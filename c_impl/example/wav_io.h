/**
 * wav_io.h - Simple WAV File I/O
 *
 * Supports PCM 16-bit and 32-bit float WAV files
 * Designed for simplicity, not comprehensive WAV support
 */

#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// WAV file information
typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int num_samples;
    int is_float;
} WavInfo;

// WAV reader handle
typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    int samples_read;
} WavReader;

// WAV writer handle
typedef struct {
    FILE* fp;
    WavInfo info;
    long data_start;
    int samples_written;
} WavWriter;

// ============================================================================
// WAV Reader
// ============================================================================

/**
 * Open WAV file for reading
 *
 * @param path Path to WAV file
 * @return Reader handle, or NULL on error
 */
static inline WavReader* wav_open_read(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;

    WavReader* r = (WavReader*)calloc(1, sizeof(WavReader));
    if (!r) {
        fclose(fp);
        return NULL;
    }
    r->fp = fp;

    // Read RIFF header
    char riff[4], wave[4];
    uint32_t chunk_size;

    if (fread(riff, 1, 4, fp) != 4) goto error;
    if (fread(&chunk_size, 4, 1, fp) != 1) goto error;
    if (fread(wave, 1, 4, fp) != 4) goto error;

    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        goto error;
    }

    // Find fmt chunk
    while (1) {
        char chunk_id[4];
        uint32_t chunk_sz;
        if (fread(chunk_id, 1, 4, fp) != 4) goto error;
        if (fread(&chunk_sz, 4, 1, fp) != 1) goto error;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format, num_channels, block_align, bits_per_sample;
            uint32_t sample_rate, byte_rate;

            if (fread(&audio_format, 2, 1, fp) != 1) goto error;
            if (fread(&num_channels, 2, 1, fp) != 1) goto error;
            if (fread(&sample_rate, 4, 1, fp) != 1) goto error;
            if (fread(&byte_rate, 4, 1, fp) != 1) goto error;
            if (fread(&block_align, 2, 1, fp) != 1) goto error;
            if (fread(&bits_per_sample, 2, 1, fp) != 1) goto error;

            r->info.sample_rate = (int)sample_rate;
            r->info.channels = (int)num_channels;
            r->info.bits_per_sample = (int)bits_per_sample;
            r->info.is_float = (audio_format == 3);

            // Skip any extra format bytes
            if (chunk_sz > 16) {
                fseek(fp, chunk_sz - 16, SEEK_CUR);
            }
            break;
        } else {
            fseek(fp, chunk_sz, SEEK_CUR);
        }
    }

    // Find data chunk
    while (1) {
        char chunk_id[4];
        uint32_t chunk_sz;
        if (fread(chunk_id, 1, 4, fp) != 4) goto error;
        if (fread(&chunk_sz, 4, 1, fp) != 1) goto error;

        if (memcmp(chunk_id, "data", 4) == 0) {
            int bytes_per_sample = r->info.bits_per_sample / 8;
            r->info.num_samples = chunk_sz / bytes_per_sample / r->info.channels;
            r->data_start = ftell(fp);
            break;
        } else {
            fseek(fp, chunk_sz, SEEK_CUR);
        }
    }

    return r;

error:
    fclose(fp);
    free(r);
    return NULL;
}

/**
 * Read samples as float (mono, first channel only)
 *
 * @param r Reader handle
 * @param buf Output buffer
 * @param n Number of samples to read
 * @return Number of samples actually read
 */
static inline int wav_read_float(WavReader* r, float* buf, int n) {
    if (!r || !buf) return 0;

    int read_count = 0;
    int channels = r->info.channels;
    int bits = r->info.bits_per_sample;

    for (int i = 0; i < n && r->samples_read < r->info.num_samples; i++) {
        if (bits == 16) {
            int16_t sample;
            if (fread(&sample, 2, 1, r->fp) != 1) break;
            buf[i] = (float)sample / 32768.0f;
            // Skip other channels
            if (channels > 1) {
                fseek(r->fp, 2 * (channels - 1), SEEK_CUR);
            }
        } else if (bits == 32 && r->info.is_float) {
            float sample;
            if (fread(&sample, 4, 1, r->fp) != 1) break;
            buf[i] = sample;
            if (channels > 1) {
                fseek(r->fp, 4 * (channels - 1), SEEK_CUR);
            }
        } else if (bits == 32) {
            int32_t sample;
            if (fread(&sample, 4, 1, r->fp) != 1) break;
            buf[i] = (float)sample / 2147483648.0f;
            if (channels > 1) {
                fseek(r->fp, 4 * (channels - 1), SEEK_CUR);
            }
        } else {
            break;  // Unsupported format
        }
        r->samples_read++;
        read_count++;
    }

    return read_count;
}

/**
 * Close WAV reader
 */
static inline void wav_close_read(WavReader* r) {
    if (r) {
        if (r->fp) fclose(r->fp);
        free(r);
    }
}

// ============================================================================
// WAV Writer
// ============================================================================

/**
 * Open WAV file for writing (16-bit PCM mono)
 *
 * @param path Path to output WAV file
 * @param sample_rate Sample rate
 * @param channels Number of channels (usually 1)
 * @return Writer handle, or NULL on error
 */
static inline WavWriter* wav_open_write(const char* path, int sample_rate, int channels) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return NULL;

    WavWriter* w = (WavWriter*)calloc(1, sizeof(WavWriter));
    if (!w) {
        fclose(fp);
        return NULL;
    }
    w->fp = fp;
    w->info.sample_rate = sample_rate;
    w->info.channels = channels;
    w->info.bits_per_sample = 16;

    // Write placeholder header (will update at close)
    uint8_t header[44] = {0};
    fwrite(header, 1, 44, fp);
    w->data_start = 44;

    return w;
}

/**
 * Write samples from float buffer
 *
 * @param w Writer handle
 * @param buf Input buffer (float samples)
 * @param n Number of samples to write
 */
static inline void wav_write_float(WavWriter* w, const float* buf, int n) {
    if (!w || !buf) return;

    for (int i = 0; i < n; i++) {
        float sample = buf[i];
        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

        int16_t s16 = (int16_t)(sample * 32767.0f);
        fwrite(&s16, 2, 1, w->fp);
        w->samples_written++;
    }
}

/**
 * Close WAV writer and finalize header
 */
static inline void wav_close_write(WavWriter* w) {
    if (!w) return;

    // Calculate sizes
    int data_size = w->samples_written * 2 * w->info.channels;
    int file_size = 36 + data_size;

    // Seek to beginning and write proper header
    fseek(w->fp, 0, SEEK_SET);

    // RIFF header
    fwrite("RIFF", 1, 4, w->fp);
    uint32_t chunk_size = file_size;
    fwrite(&chunk_size, 4, 1, w->fp);
    fwrite("WAVE", 1, 4, w->fp);

    // fmt chunk
    fwrite("fmt ", 1, 4, w->fp);
    uint32_t subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, w->fp);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, w->fp);
    uint16_t num_channels = w->info.channels;
    fwrite(&num_channels, 2, 1, w->fp);
    uint32_t sample_rate = w->info.sample_rate;
    fwrite(&sample_rate, 4, 1, w->fp);
    uint32_t byte_rate = sample_rate * num_channels * 2;
    fwrite(&byte_rate, 4, 1, w->fp);
    uint16_t block_align = num_channels * 2;
    fwrite(&block_align, 2, 1, w->fp);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, 2, 1, w->fp);

    // data chunk header
    fwrite("data", 1, 4, w->fp);
    uint32_t data_sz = data_size;
    fwrite(&data_sz, 4, 1, w->fp);

    fclose(w->fp);
    free(w);
}

#endif // WAV_IO_H
