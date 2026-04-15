// iio_client.cpp - LSM6DSOX Multicast Client
// NY: Tid beregnes fra differansen mellom påfølgende batch_ts fra sensor
// - ingen lokal fs-kalibrering
// - dt_per_sample = (batch_ts[n] - batch_ts[n-1]) / NUM_SAMPLES
// NY: Støtte for flere CSV-filer per kjøring (_1, _2, _3, ...)
//     - hver fil inneholder samme antall pakker som originalt (~1 minutt)
//     - antall_sessions styrer hvor mange filer som lages

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <limits>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

using namespace std;

#define PORT        8080
#define MCAST_IP    "239.192.200.10"
#define LOCAL_IP    "192.168.200.100"  // Endre til din lokale IP
#define NUM_SAMPLES 64

#pragma pack(push, 1)
struct Sample {
    int16_t x;
    int16_t y;
    int16_t z;
    // ENDRET: timestamp fjernet fra Sample - bruker batch_ts fra header
};

struct PacketHeader {
    uint32_t sequence;
    uint32_t sample_count;
    int64_t  batch_ts;   // NY: rå batch-timestamp fra sensor
};

struct Packet {
    PacketHeader hdr;
    Sample       samples[NUM_SAMPLES];
};
#pragma pack(pop)

static volatile bool running = true;
void signal_handler(int) { running = false; }

// NY: Antall CSV-filer/sesjoner å samle inn (f.eks. 5 = ca 5 minutter)
static const int ANTALL_SESSIONS = 5;

// Hardkodet antall pakker per sesjon - må matche serveren (SESSION_DURATION=60s ved 833Hz nominelt)
// Server bruker tidsstyring (60 sek), klient teller pakker.
// Ved ~842 Hz faktisk: ~789 pakker/min. Sett til 780 for å stoppe litt før server,
// slik at klienten aldri venter på pakker som ikke kommer.
// Juster om server produserer konsekvent mer eller mindre enn dette.
static const uint32_t MAX_PACKETS_PER_SESSION = 780;

int main()
{
    cout << "LSM6DSOX Multicast Client\n";
    cout << "=========================\n\n";
    cout << "Antall sesjoner: " << ANTALL_SESSIONS << "\n";
    cout << "Pakker per sesjon: " << MAX_PACKETS_PER_SESSION
         << "  (" << MAX_PACKETS_PER_SESSION * NUM_SAMPLES << " samples)\n\n";

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

    // fallback dt_ns brukes kun for første batch (ingen forrige ts ennå)
    // 833 Hz => ~1201923 ns per sample
    const double FALLBACK_FS = 833.0;

    // holder styr på forrige batch_ts for å beregne dt - persistent på tvers av sesjoner
    int64_t  prev_batch_ts    = numeric_limits<int64_t>::min();
    int64_t  dt_per_sample_ns = (int64_t)(1e9 / FALLBACK_FS);
    uint32_t last_seq         = numeric_limits<uint32_t>::max();

    // Generer mappestruktur: data/MM/DD/HHMMSS/
    time_t now = time(nullptr);
    tm* t = localtime(&now);

    char dir_month[32], dir_date[32], dir_time[32], dir_full[128];
    strftime(dir_month, sizeof(dir_month), "data/%m",          t);  // data/03
    strftime(dir_date,  sizeof(dir_date),  "data/%m/%d",       t);  // data/03/09
    strftime(dir_time,  sizeof(dir_time),  "data/%m/%d/%H%M%S", t); // data/03/09/120000

    // Opprett mapper nivå for nivå
    mkdir(dir_month, 0755);
    mkdir(dir_date,  0755);
    mkdir(dir_time,  0755);

    // base_filename peker inn i den nye mappen
    char base_filename[128];
    snprintf(base_filename, sizeof(base_filename), "%s/iio_data", dir_time);

    cout << "Mappe: " << dir_time << "\n";
    cout << "Waiting for packets... (Ctrl+C to stop)\n\n";

    // Ytre løkke - én iterasjon per sesjon/CSV-fil
    for (int session = 1; session <= ANTALL_SESSIONS && running; session++)
    {
        // Generer filnavn med _N suffiks
        char filename[160];
        snprintf(filename, sizeof(filename), "%s_%d.csv", base_filename, session);

        ofstream csvfile(filename, ios::out);
        if (!csvfile) {
            cerr << "ERROR: Failed to open CSV file: " << filename << "\n";
            break;
        }

        // Pen formatering for tid
        csvfile << fixed << setprecision(9);

        // Header
        csvfile << "sequence,sample,x,y,z,time_s\n";
        cout << "\n--- Sesjon " << session << "/" << ANTALL_SESSIONS
             << " -> " << filename << " ---\n";

        int64_t  start_ts      = numeric_limits<int64_t>::min();
        uint32_t total_packets = 0;
        uint32_t total_samples = 0;
        uint32_t lost_packets  = 0;

        auto t0 = chrono::steady_clock::now();

        // Stopper etter nøyaktig MAX_PACKETS_PER_SESSION pakker - fast, ingen dynamisk justering
        while (running && total_packets < MAX_PACKETS_PER_SESSION)
        {
            Packet pkt{};
            sockaddr_storage their_addr{};
            socklen_t addr_len = sizeof(their_addr);

            ssize_t n = recvfrom(sockfd, &pkt, sizeof(pkt), 0,
                                 (sockaddr*)&their_addr, &addr_len);

            if (n < 0) { if (running) perror("ERROR: recvfrom"); goto cleanup; }
            if (n != (ssize_t)sizeof(Packet)) {
                cerr << "WARNING: Uventet pakkestørrelse: " << n
                     << " bytes, forventet " << sizeof(Packet) << "\n";
                continue;
            }

            // Sjekk for tapte pakker
            if (last_seq != numeric_limits<uint32_t>::max()) {
                uint32_t gap = pkt.hdr.sequence - (last_seq + 1);
                if (gap > 0) {
                    lost_packets += gap;
                    cerr << "WARNING: Tapte " << gap << " pakke(r) "
                         << "(seq=" << last_seq + 1 << " til " << pkt.hdr.sequence - 1 << ")\n";
                }
            }
            last_seq = pkt.hdr.sequence;

            int64_t batch_ts = pkt.hdr.batch_ts;

            // Sett start_ts fra første gyldige base_ts (per sesjon)
            // base_ts = batch_ts - 63*dt, dvs. tidspunktet til sample 0 i første batch
            // slik at time_s alltid starter på 0.0 for første sample
            if (start_ts == numeric_limits<int64_t>::min() && batch_ts != 0)
                start_ts = batch_ts - (int64_t)(NUM_SAMPLES - 1) * dt_per_sample_ns;

            // Beregn dt_per_sample fra differansen mellom to påfølgende batch_ts
            // Første batch bruker fallback. Deretter brukes faktisk dt fra sensor.
            // prev_batch_ts er persistent på tvers av sesjoner for jevn overgang.
            // MERK: dt brukes kun til tidsutregning i CSV, ikke til å styre antall pakker.
            if (prev_batch_ts != numeric_limits<int64_t>::min() && batch_ts != 0) {
                int64_t dt_total = batch_ts - prev_batch_ts;
                if (dt_total > 0)
                    dt_per_sample_ns = dt_total / NUM_SAMPLES;
            }
            prev_batch_ts = batch_ts;

            // S0 starter ved forrige batch_ts (ikke denne), siden batch_ts
            // er tidspunktet da de 64 samplingene var FERDIG lest.
            // => S0 = batch_ts - 63*dt, S1 = batch_ts - 62*dt, ..., S63 = batch_ts
            int64_t base_ts = (batch_ts != 0)
                              ? batch_ts - (int64_t)(NUM_SAMPLES - 1) * dt_per_sample_ns
                              : 0;

            for (unsigned int i = 0; i < pkt.hdr.sample_count; i++) {
                int64_t sample_ts = base_ts + (int64_t)i * dt_per_sample_ns;

                double time_sec = 0.0;
                if (start_ts != numeric_limits<int64_t>::min() && sample_ts != 0)
                    time_sec = (sample_ts - start_ts) / 1e9;

                csvfile << pkt.hdr.sequence   << ","
                    << (i + 1)            << ","
                    << pkt.samples[i].x   << ","
                    << pkt.samples[i].y   << ","
                    << pkt.samples[i].z   << ","
                    << time_sec           << "\n";
            }

            total_packets++;
            total_samples += pkt.hdr.sample_count;

            cout << "Pakke [" << pkt.hdr.sequence << "]"
                 << "  samples: " << total_samples
                 << "  dt=" << dt_per_sample_ns / 1000 << " µs"
                 << "  -> " << filename << "\r" << flush;
        }

        auto t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;

        cout << "\n\nSesjon " << session << " ferdig.\n";
        cout << "Totalt mottatt: " << total_packets << " pakker, "
             << total_samples << " samples\n";
        cout << "Tapte pakker:   " << lost_packets << "\n";
        cout << "Siste dt:       " << dt_per_sample_ns / 1000 << " µs/sample"
             << "  (~" << 1e9 / dt_per_sample_ns << " Hz)\n";
        cout << "Loop-tid: " << elapsed.count() << " sek\n";
        cout << "Data lagret i: " << filename << "\n";

        csvfile.close();
    } // end session loop

    cout << "\nStopper klient...\n";

cleanup:
    setsockopt(sockfd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
    close(sockfd);
    return 0;
}