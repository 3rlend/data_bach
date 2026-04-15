// iio_client.cpp - LSM6DSOX Multicast Client (kontinuerlig / tidsstyrt)
// Bruker skriver 'k' for kontinuerlig modus eller antall minutter.
// Splitter CSV-filer etter fast antall pakker (_1, _2, _3, ...).
// 780 pakker ≈ 60 sek ved ~833 Hz (samme logikk som originalen).
// time_s resetter til 0.0 for hver ny fil.
// Kompatibel med iio_server_continuous.cpp.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <limits>
#include <chrono>
#include <ctime>
#include <string>
#include <sys/stat.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

using namespace std;

#define PORT        8080
#define MCAST_IP    "239.192.200.10"
#define LOCAL_IP    "192.168.200.100"
#define NUM_SAMPLES 64

// Antall pakker per fil (~60 sek ved ~833 Hz: 833*60/64 ≈ 780)
static const uint32_t PACKETS_PER_FILE = 780;

#pragma pack(push, 1)
struct Sample {
    int16_t x;
    int16_t y;
    int16_t z;
};

struct PacketHeader {
    uint32_t sequence;
    uint32_t sample_count;
    int64_t  batch_ts;
};

struct Packet {
    PacketHeader hdr;
    Sample       samples[NUM_SAMPLES];
};
#pragma pack(pop)

static volatile bool running = true;
void signal_handler(int) { running = false; }

int main()
{
    cout << "LSM6DSOX Multicast Client\n";
    cout << "=========================\n\n";

    // --- Brukervalg: kontinuerlig eller antall minutter ---
    bool continuous = false;
    double duration_min = 0.0;

    cout << "Skriv 'k' for kontinuerlig modus, eller antall minutter: ";
    string input;
    getline(cin, input);

    if (input == "k" || input == "K") {
        continuous = true;
        cout << "Kontinuerlig modus - stopp med Ctrl+C\n";
    } else {
        try {
            duration_min = stod(input);
            if (duration_min <= 0) {
                cerr << "Ugyldig verdi. Avslutter.\n";
                return 1;
            }
            cout << "Samler data i " << duration_min << " minutt(er)\n";
        } catch (...) {
            cerr << "Ugyldig input. Skriv 'k' eller et tall (f.eks. 5). Avslutter.\n";
            return 1;
        }
    }

    // Beregn totalt antall filer (kun brukt i tidsstyrt modus)
    int total_files = 0;
    if (!continuous) {
        total_files = (int)(duration_min + 0.999);  // rund opp: 5 min -> 5 filer
        cout << "Antall filer: " << total_files << " (à "
             << PACKETS_PER_FILE << " pakker / " << PACKETS_PER_FILE * NUM_SAMPLES
             << " samples)\n";
    }
    cout << "\n";

    // --- Socket-oppsett ---
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("ERROR: Failed to create socket"); return 1; }

    int reuse = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("WARNING: setsockopt(SO_REUSEADDR) failed");
    }

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockfd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("ERROR: Bind failed");
        close(sockfd);
        return 1;
    }
    cout << "Bound to port " << PORT << "\n";

    ip_mreq mreq{};
    inet_pton(AF_INET, MCAST_IP, &mreq.imr_multiaddr);
    inet_pton(AF_INET, LOCAL_IP, &mreq.imr_interface);
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror("ERROR: Multicast join failed");
        close(sockfd);
        return 1;
    }
    cout << "Joined multicast group " << MCAST_IP << "\n";

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // --- Generer mappestruktur ---
    time_t now = time(nullptr);
    tm* t = localtime(&now);

    char dir_month[32], dir_date[32], dir_time[32];
    strftime(dir_month, sizeof(dir_month), "data/%m",          t);
    strftime(dir_date,  sizeof(dir_date),  "data/%m/%d",       t);
    strftime(dir_time,  sizeof(dir_time),  "data/%m/%d/%H%M%S", t);

    mkdir("data",    0755);
    mkdir(dir_month, 0755);
    mkdir(dir_date,  0755);
    mkdir(dir_time,  0755);

    char base_filename[160];
    snprintf(base_filename, sizeof(base_filename), "%s/iio_data", dir_time);

    cout << "Mappe: " << dir_time << "\n";
    cout << "Pakker per fil: " << PACKETS_PER_FILE
         << "  (" << PACKETS_PER_FILE * NUM_SAMPLES << " samples)\n";
    cout << "Waiting for packets... (Ctrl+C to stop)\n\n";

    // --- Variabler for tidsberegning (persistent på tvers av filer) ---
    const double FALLBACK_FS = 833.0;
    int64_t  prev_batch_ts    = numeric_limits<int64_t>::min();
    int64_t  dt_per_sample_ns = (int64_t)(1e9 / FALLBACK_FS);
    uint32_t last_seq         = numeric_limits<uint32_t>::max();

    uint32_t total_packets = 0;
    uint64_t total_samples = 0;
    uint32_t lost_packets  = 0;

    auto t0 = chrono::steady_clock::now();

    // --- Filsplitting-variabler ---
    int file_number = 0;
    ofstream csvfile;
    int64_t file_start_ts = numeric_limits<int64_t>::min();
    uint32_t file_packets = 0;
    char current_filename[192] = {};

    // Lambda for å åpne ny fil
    auto open_new_file = [&]() {
        if (csvfile.is_open()) {
            csvfile.close();
            cout << "\n  -> " << current_filename << " ferdig ("
                 << file_packets << " pakker)\n";
        }

        file_number++;
        snprintf(current_filename, sizeof(current_filename),
                 "%s_%d.csv", base_filename, file_number);

        csvfile.open(current_filename, ios::out);
        if (!csvfile) {
            cerr << "ERROR: Failed to open CSV file: " << current_filename << "\n";
            running = false;
            return;
        }

        csvfile << fixed << setprecision(9);
        csvfile << "sequence,sample,x,y,z,time_s\n";

        file_start_ts = numeric_limits<int64_t>::min();
        file_packets  = 0;

        cout << "\n--- Fil " << file_number;
        if (!continuous) cout << "/" << total_files;
        cout << ": " << current_filename << " ---\n";
    };

    // Åpne første fil
    open_new_file();

    // === HOVEDLØKKE ===
    while (running)
    {
        // I tidsstyrt modus: stopp når vi har nok filer
        if (!continuous && file_number > total_files)
            break;

        // Sjekk om filen er full -> åpne ny
        if (file_packets >= PACKETS_PER_FILE) {
            // I tidsstyrt modus: sjekk om vi allerede har nok filer
            if (!continuous && file_number >= total_files)
                break;
            open_new_file();
        }

        if (!running) break;

        Packet pkt{};
        sockaddr_storage their_addr{};
        socklen_t addr_len = sizeof(their_addr);

        ssize_t n = recvfrom(sockfd, &pkt, sizeof(pkt), 0,
                             (sockaddr*)&their_addr, &addr_len);

        if (n < 0) { if (running) perror("ERROR: recvfrom"); break; }
        if (n != (ssize_t)sizeof(Packet)) {
            cerr << "WARNING: Uventet pakkestørrelse: " << n
                 << " bytes, forventet " << sizeof(Packet) << "\n";
            continue;
        }

        // Sjekk for tapte pakker
        if (last_seq != numeric_limits<uint32_t>::max()) {
            uint32_t gap = pkt.hdr.sequence - (last_seq + 1);
            if (gap > 0 && gap < 10000) {
                lost_packets += gap;
                cerr << "WARNING: Tapte " << gap << " pakke(r) "
                     << "(seq=" << last_seq + 1 << " til " << pkt.hdr.sequence - 1 << ")\n";
            }
        }
        last_seq = pkt.hdr.sequence;

        int64_t batch_ts = pkt.hdr.batch_ts;

        // Sett file_start_ts fra første gyldige batch i denne filen
        if (file_start_ts == numeric_limits<int64_t>::min() && batch_ts != 0)
            file_start_ts = batch_ts - (int64_t)(NUM_SAMPLES - 1) * dt_per_sample_ns;

        // Beregn dt_per_sample fra differansen mellom påfølgende batch_ts
        if (prev_batch_ts != numeric_limits<int64_t>::min() && batch_ts != 0) {
            int64_t dt_total = batch_ts - prev_batch_ts;
            if (dt_total > 0)
                dt_per_sample_ns = dt_total / NUM_SAMPLES;
        }
        prev_batch_ts = batch_ts;

        // Beregn base_ts for denne batchen
        int64_t base_ts = (batch_ts != 0)
                          ? batch_ts - (int64_t)(NUM_SAMPLES - 1) * dt_per_sample_ns
                          : 0;

        for (unsigned int i = 0; i < pkt.hdr.sample_count; i++) {
            int64_t sample_ts = base_ts + (int64_t)i * dt_per_sample_ns;

            double time_sec = 0.0;
            if (file_start_ts != numeric_limits<int64_t>::min() && sample_ts != 0)
                time_sec = (sample_ts - file_start_ts) / 1e9;

            csvfile << pkt.hdr.sequence   << ","
                    << (i + 1)            << ","
                    << pkt.samples[i].x   << ","
                    << pkt.samples[i].y   << ","
                    << pkt.samples[i].z   << ","
                    << time_sec           << "\n";
        }

        total_packets++;
        file_packets++;
        total_samples += pkt.hdr.sample_count;

        // Statuslinje
        double elapsed_s = chrono::duration<double>(
            chrono::steady_clock::now() - t0).count();

        if (!continuous) {
            cout << "\rfil " << file_number << "/" << total_files
                 << "  pkt=" << file_packets << "/" << PACKETS_PER_FILE
                 << "  seq=" << pkt.hdr.sequence
                 << "  tot=" << total_samples
                 << "  dt=" << dt_per_sample_ns / 1000 << " µs"
                 << "       " << flush;
        } else {
            cout << "\rfil=" << file_number
                 << "  pkt=" << file_packets << "/" << PACKETS_PER_FILE
                 << "  seq=" << pkt.hdr.sequence
                 << "  tot=" << total_samples
                 << "  dt=" << dt_per_sample_ns / 1000 << " µs"
                 << "  tid=" << fixed << setprecision(1) << elapsed_s << " s"
                 << "       " << flush;
        }
    }

    // Lukk siste fil
    if (csvfile.is_open()) {
        csvfile.close();
        cout << "\n  -> " << current_filename << " ferdig ("
             << file_packets << " pakker)\n";
    }

    // --- Oppsummering ---
    double total_s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    cout << "\n--- Klient stoppet ---\n";
    cout << "Totalt mottatt: " << total_packets << " pakker, "
         << total_samples << " samples\n";
    cout << "Filer lagret:   " << file_number << " stk i " << dir_time << "/\n";
    cout << "Tapte pakker:   " << lost_packets << "\n";
    cout << "Siste dt:       " << dt_per_sample_ns / 1000 << " µs/sample"
         << "  (~" << fixed << setprecision(0) << 1e9 / dt_per_sample_ns << " Hz)\n";
    cout << "Kjøretid:       " << setprecision(1) << total_s << " sek\n";

    setsockopt(sockfd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
    close(sockfd);
    return 0;
}