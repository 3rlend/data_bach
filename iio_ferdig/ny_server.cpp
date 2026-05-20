// iio_server.cpp - LSM6DSOX accelerometer multicast server (kontinuerlig modus)
// Bygger på logikken fra iio_test.cpp, men sender NUM_SAMPLES per UDP-pakke.
// Server sender kun batch_ts i headeren - klienten gjør all tidsutregning.
// Klienten styrer varighet og filhåndtering selv.

#include <iostream>
#include <iomanip>r ---
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <csignal>

#include <iio.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

#define PORT        8080
#define MCAST_IP    "239.192.200.10"
#define NUM_SAMPLES 64

// Signal-håndtering for clean shutdown
static volatile bool running = true;
void signal_handler(int) { running = false; }

#pragma pack(push, 1) //Fjerne unødvendig filler mellom bytsene
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
    cout << "LSM6DSOX Multicast Server (kontinuerlig modus)\n";
    cout << "===============================================\n\n";

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

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

    // 4b) Les tilbake faktisk fs
    double fs = 833.0;
    char freq_str[32]{};
    if (iio_device_attr_read(dev, "sampling_frequency", freq_str, sizeof(freq_str)) > 0) {
        fs = strtod(freq_str, nullptr);
        cout << "Faktisk fs = " << fs << " Hz\n";
    }

    // 5) Set full-scale (±8g)
    ret = iio_device_attr_write(dev, "in_accel_scale", "0.002392822"); // Mulige verdier: 2g: 0.000598205 4g: 0.001196411 8g: 0.002392822 16g: 0.004785645
    if (ret < 0) cerr << "WARNING: Failed to set full-scale: " << ret << "\n";
    else         cout << "Set full-scale to ±8g (0.002392822)\n";

    // 6) Find channels
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

    // 7) Enable channels
    iio_channel_enable(ch_x);
    iio_channel_enable(ch_y);
    iio_channel_enable(ch_z);
    if (ch_time) iio_channel_enable(ch_time);

    // 8) Create buffer
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

    sockaddr_in mcast_addr{};
    mcast_addr.sin_family = AF_INET;
    mcast_addr.sin_port   = htons(PORT);
    inet_pton(AF_INET, MCAST_IP, &mcast_addr.sin_addr);

    cout << "Streaming " << NUM_SAMPLES << " samples per UDP packet to "
         << MCAST_IP << ":" << PORT << "\n";
    cout << "Kjører kontinuerlig - stopp med Ctrl+C\n\n";

    uint32_t sequence = 0;

    // Midlertidige arrays
    vector<int16_t> x(NUM_SAMPLES), y(NUM_SAMPLES), z(NUM_SAMPLES);

    // Payload buffer: header + samples
    vector<uint8_t> payload(sizeof(PacketHeader) + NUM_SAMPLES * sizeof(Sample));

    auto t_start = chrono::steady_clock::now();
    uint64_t total_loops = 0;

    // === HOVEDLØKKE - kjører til Ctrl+C ===
    while (running)
    {
        // Hent nye samples fra sensor
        auto t_before = chrono::steady_clock::now();
        ret = iio_buffer_refill(buf);
        auto t_after  = chrono::steady_clock::now();

        if (ret < 0) {
            if (running) cerr << "ERROR: iio_buffer_refill failed: " << ret << "\n";
            break;
        }

        // Statuslogg hvert 100. refill
        if (total_loops % 100 == 0) {
            double ms = chrono::duration<double>(t_after - t_before).count() * 1000;
            double elapsed_s = chrono::duration<double>(t_after - t_start).count();
            cout << "\r[" << fixed << setprecision(1) << elapsed_s << " s] "
                 << "seq=" << sequence
                 << "  refill=" << setprecision(2) << ms << " ms"
                 << "  (forventet " << setprecision(2) << (64.0/fs)*1000 << " ms)"
                 << "       " << flush;
        }

        // Les kanaldata
        auto bx = iio_channel_read(ch_x, buf, x.data(), NUM_SAMPLES * sizeof(int16_t));
        auto by = iio_channel_read(ch_y, buf, y.data(), NUM_SAMPLES * sizeof(int16_t));
        auto bz = iio_channel_read(ch_z, buf, z.data(), NUM_SAMPLES * sizeof(int16_t));

        // Les batch timestamp (kun 1 per refill)
        int64_t batch_ts = 0;
        ssize_t bt = 0;
        if (ch_time) {
            int64_t one = 0;
            bt = iio_channel_read(ch_time, buf, &one, sizeof(one));
            if (bt == (ssize_t)sizeof(one))
                batch_ts = one;
        }

        // Short read-sjekk
        if (bx != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
            by != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
            bz != (ssize_t)(NUM_SAMPLES * sizeof(int16_t)) ||
            (ch_time && bt != (ssize_t)sizeof(int64_t))) {
            cerr << "\nSHORT READ: bx=" << bx
                 << " by=" << by
                 << " bz=" << bz
                 << " bt=" << bt << "\n";
        }

        // Bygg header
        PacketHeader hdr{};
        hdr.sequence     = sequence++;
        hdr.sample_count = NUM_SAMPLES;
        hdr.batch_ts     = batch_ts;

        std::memcpy(payload.data(), &hdr, sizeof(hdr));

        // Fyll inn samples
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
            if (running) cerr << "\nERROR: sendto failed\n";
            break;
        }

        total_loops++;
    }

    // Oppsummering ved avslutning i terminal
    auto t_end = chrono::steady_clock::now();
    double total_s = chrono::duration<double>(t_end - t_start).count();

    cout << "\n\n--- Server stoppet ---\n";
    cout << "Totalt sendt:  " << sequence << " pakker ("
         << (uint64_t)sequence * NUM_SAMPLES << " samples)\n";
    cout << "Kjøretid:      " << fixed << setprecision(1) << total_s << " sek\n";
    cout << "Gjennomsnitt:  " << setprecision(1) << (sequence / total_s) << " pakker/sek"
         << "  (~" << setprecision(0) << (sequence * NUM_SAMPLES / total_s) << " Hz)\n";

    //iio sluttoprasjoner
    iio_buffer_destroy(buf);
    iio_context_destroy(ctx);
    close(sockfd);
    return 0;
}