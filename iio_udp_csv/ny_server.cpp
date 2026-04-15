// iio_server.cpp - LSM6DSOX accelerometer multicast server (correct sampling)
// Bygger på logikken fra iio_test.cpp, men sender NUM_SAMPLES per UDP-pakke.
// Server sender kun batch_ts i headeren - klienten gjør all tidsutregning
// NY: Støtte for flere sesjoner - sender i f.eks. 10 minutter totalt
//     - antall_sessions styrer antall 1-minutts sesjoner

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>    // trengs for llround()
#include <cstdlib>  // trengs for strtod()
#include <chrono>   // trengs for buffer tid

#include <iio.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

#define PORT        8080
#define MCAST_IP    "239.192.200.10"
#define NUM_SAMPLES 64

// NY: Antall sesjoner à ~1 minutt (f.eks. 5 = ca 5 minutter totalt)
static const int ANTALL_SESSIONS = 5;

// Ett sample (råverdier) - ENDRET: timestamp fjernet, batch_ts ligger nå i headeren
#pragma pack(push, 1)
struct Sample {
    int16_t x;
    int16_t y;
    int16_t z;
};

// Liten header foran hver UDP-pakke
struct PacketHeader {
    uint32_t sequence;     // øker med 1 per pakke
    uint32_t sample_count; // hvor mange Sample som følger
    int64_t  batch_ts;     // NY: rå batch-timestamp fra sensor (nanosekunder)
};
#pragma pack(pop)

static void list_device_attributes(iio_device* dev)
{
    cout << "\nDevice attributes:\n";
    unsigned int numAttrs = iio_device_get_attrs_count(dev);
    for (unsigned int i = 0; i < numAttrs; i++) {
        const char* attrName = iio_device_get_attr(dev, i);
        char attrValue[256]{};
        if (iio_device_attr_read(dev, attrName, attrValue, sizeof(attrValue)) > 0)
            cout << "  " << attrName << " = " << attrValue;
    }
    cout << "\n";
}

int main()
{
    cout << "LSM6DSOX Multicast Server (correct sampling)\n";
    cout << "===========================================\n\n";

    // 1) Create IIO context
    iio_context* ctx = iio_create_default_context();
    if (!ctx) { cerr << "ERROR: Failed to create IIO context\n"; return 1; }
    cout << "IIO context created successfully\n\n";

    // 2) Find device
    iio_device* dev = iio_context_find_device(ctx, "lsm6dsox_accel");
    if (!dev) {
        cerr << "ERROR: lsm6dsox_accel device not found\n";
        iio_context_destroy(ctx);
        return 1;
    }
    cout << "Found device: lsm6dsox_accel\n";

    // 3) List attributes
    list_device_attributes(dev);

    // 4) Set sampling frequency
    int ret = iio_device_attr_write(dev, "sampling_frequency", "833");
    if (ret < 0) cerr << "WARNING: Failed to set sampling frequency: " << ret << "\n";
    else         cout << "Set sampling frequency to 833 Hz\n";

    // 4b) Les tilbake faktisk fs og bruk den for antall_loops
    double fs = 833.0;
    char freq_str[32]{};
    if (iio_device_attr_read(dev, "sampling_frequency", freq_str, sizeof(freq_str)) > 0) {
        fs = strtod(freq_str, nullptr);
        cout << "Faktisk fs = " << fs << " Hz (brukes for loop-telling)\n";
    }
    const int antall_loops = (int)(fs * 60) / NUM_SAMPLES; // bruk faktisk fs for loop-telling

    // dt_ns ikke lenger nødvendig på server - klienten regner ut dt selv fra batch_ts differanse

    // 5) Set full-scale (±4g = 0.000122 g/LSB)
    ret = iio_device_attr_write(dev, "in_accel_scale", "0.001196411"); // Mulige verdier: 2g: 0.000598205 4g: 0.001196411 8g: 0.002392822 16g: 0.004785645
    if (ret < 0) cerr << "WARNING: Failed to set full-scale: " << ret << "\n";
    else         cout << "Set full-scale to ±4g (0.000122 g/LSB)\n";

    // // 6) Set LPF2 filter (0 = off, 1 = on) Funker ikke
    // ret = iio_device_attr_write(dev, "filter_low_pass_3db_frequency", "0");
    // if (ret < 0)
    //     cerr << "WARNING: Failed to set LPF2: " << ret << "\n";
    // else
    //     cout << "Set LPF2 filter: off\n";

    // 7) Find channels
    iio_channel* ch_x    = iio_device_find_channel(dev, "accel_x",   false);
    iio_channel* ch_y    = iio_device_find_channel(dev, "accel_y",   false);
    iio_channel* ch_z    = iio_device_find_channel(dev, "accel_z",   false);
    iio_channel* ch_time = iio_device_find_channel(dev, "timestamp", false);

    if (!ch_x || !ch_y || !ch_z) {
        cerr << "ERROR: Failed to find accel channels\n";
        iio_context_destroy(ctx);
        return 1;
    }
    cout << "Found channels: accel_x, accel_y, accel_z\n";
    if (ch_time) cout << "Found channel: timestamp\n";
    cout << "\n";

    // 8) Enable channels
    iio_channel_enable(ch_x);
    iio_channel_enable(ch_y);
    iio_channel_enable(ch_z);
    if (ch_time) iio_channel_enable(ch_time);

    // 9) Create buffer for NUM_SAMPLES (one refill => block of NUM_SAMPLES)
    iio_buffer* buf = iio_device_create_buffer(dev, NUM_SAMPLES, false);
    if (!buf) {
        cerr << "ERROR: Failed to create buffer\n";
        iio_context_destroy(ctx);
        return 1;
    }

    // ---- UDP multicast socket ----
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        cerr << "ERROR: Failed to create socket\n";
        iio_buffer_destroy(buf);
        iio_context_destroy(ctx);
        return 1;
    }

    int ttl = 1;
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
        cerr << "WARNING: setsockopt(IP_MULTICAST_TTL) failed\n";
    }

    // inet_pton() — anbefalt av Beej kap 3.4 (inet_addr() er foreldet)
    sockaddr_in mcast_addr{};
    mcast_addr.sin_family = AF_INET;
    mcast_addr.sin_port   = htons(PORT);
    inet_pton(AF_INET, MCAST_IP, &mcast_addr.sin_addr);

    cout << "Streaming " << NUM_SAMPLES << " samples per UDP packet to "
         << MCAST_IP << ":" << PORT << " ...\n";
    cout << "Antall sesjoner: " << ANTALL_SESSIONS
         << "  (~" << ANTALL_SESSIONS << " minutt(er) totalt)\n\n";

    uint32_t sequence = 0; // NY: sequence er persistent på tvers av sesjoner

    // Midlertidige arrays for å lese hele blokka per kanal
    vector<int16_t> x(NUM_SAMPLES), y(NUM_SAMPLES), z(NUM_SAMPLES);
    vector<int64_t> ts;
    if (ch_time) ts.resize(NUM_SAMPLES);

    // Payload buffer: header + samples
    // Payload er nå mindre siden Sample ikke lenger har timestamp-felt
    vector<uint8_t> payload(sizeof(PacketHeader) + NUM_SAMPLES * sizeof(Sample));

    // NY: Ytre løkke - én iterasjon per sesjon (~1 minutt)
    for (int session = 1; session <= ANTALL_SESSIONS; session++)
    {
        cout << "\n--- Sesjon " << session << "/" << ANTALL_SESSIONS << " ---\n";

        int loops = 0;
        auto t0 = chrono::steady_clock::now();

        while (loops < antall_loops)
        {
            // Hent nye samples fra sensor — mål hvor lang tid refill tar
            auto t_before = chrono::steady_clock::now();
            ret = iio_buffer_refill(buf);
            auto t_after  = chrono::steady_clock::now();

            if (ret < 0) { cerr << "ERROR: iio_buffer_refill failed: " << ret << "\n"; goto cleanup; }

            if (loops % 100 == 0) {
                double ms = chrono::duration<double>(t_after - t_before).count() * 1000;
                cout << "\n[sesjon " << session << " loop " << loops << "] refill = " << ms
                     << " ms (forventet " << (64.0/fs)*1000 << " ms)\n";
            }

            // Les kanaldata inn i arrayene
            auto bx = iio_channel_read(ch_x, buf, x.data(), NUM_SAMPLES * sizeof(int16_t));
            auto by = iio_channel_read(ch_y, buf, y.data(), NUM_SAMPLES * sizeof(int16_t));
            auto bz = iio_channel_read(ch_z, buf, z.data(), NUM_SAMPLES * sizeof(int16_t));

            // Timestamp-kanalen i IIO gir kun 1 timestamp per refill (batch), ikke 64.
            // Les KUN én int64_t og send den rått i headeren - klienten sprer den selv.
            int64_t batch_ts = 0;
            ssize_t bt = 0;
            if (ch_time) {
                int64_t one = 0;
                bt = iio_channel_read(ch_time, buf, &one, sizeof(one));
                if (bt == (ssize_t)sizeof(one)) {
                    batch_ts = one;
                } else {
                    batch_ts = 0; // ingen gyldig timestamp i denne refill'en
                }
            }

            // Short read-sjekk: timestamp forventes nå 8 bytes (1 int64_t), ikke NUM_SAMPLES*8.
            if (bx != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
                by != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
                bz != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
                (ch_time && bt != (ssize_t)sizeof(int64_t))) {
                cerr << "SHORT READ: bx=" << bx
                     << " by=" << by
                     << " bz=" << bz
                     << " bt=" << bt << "\n";
            }

            // Bygg header - batch_ts legges her, ingen spreading til samples lenger
            PacketHeader hdr{};
            hdr.sequence     = sequence++;
            hdr.sample_count = NUM_SAMPLES;
            hdr.batch_ts     = batch_ts;

            // Kopier header inn i payload
            std::memcpy(payload.data(), &hdr, sizeof(hdr));

            // Pek til Sample-området etter headeren - kun x, y, z
            Sample* out = reinterpret_cast<Sample*>(payload.data() + sizeof(PacketHeader));
            for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
                out[i].x = x[i];
                out[i].y = y[i];
                out[i].z = z[i];
            }

            // Send UDP multicast
            ssize_t sent = sendto(sockfd,
                                  payload.data(),
                                  payload.size(),
                                  0,
                                  (sockaddr*)&mcast_addr,
                                  sizeof(mcast_addr));
            if (sent < 0) {
                cerr << "ERROR: sendto failed\n";
                goto cleanup;
            }

            loops++;
        }

        auto t1 = chrono::steady_clock::now();
        chrono::duration<double> elapsed = t1 - t0;
        cout << "\nSesjon " << session << " ferdig."
             << "  loops=" << loops
             << "  tid=" << elapsed.count() << " sek"
             << "  seq=" << sequence << "\n";

    } // end session loop

    cout << "\nAlle sesjoner fullført. Done.\n";

cleanup:
    iio_buffer_destroy(buf);
    iio_context_destroy(ctx);
    close(sockfd);
    return 0;
}
