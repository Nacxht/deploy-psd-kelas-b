from moviepy import AudioFileClip

def ubah_wav_ke_mp3(file_wav, file_mp3):
    try:
        print(f"Sedang mengonversi {file_wav}...")
        # Membaca file wav
        audio = AudioFileClip(file_wav)
        
        # Menyimpannya menjadi format MP3
        audio.write_audiofile(file_mp3)
        audio.close()
        
        print(f"Berhasil! File tersimpan sebagai: {file_mp3}\n")
    except Exception as e:
        print(f"Gagal mengonversi: {e}")

# Contoh penggunaan untuk satu file
ubah_wav_ke_mp3("bifore-knalpot_standart/standart_8.wav", "after-knalpot_standart/standart_8.mp3")

# Contoh penggunaan jika ingin menggunakan looping (perulangan)
# for i in range(7):
#     ubah_wav_ke_mp3(f"bifore-knalpot_standart/standart_{i+1}.wav", f"after-knalpot_standart/standart_{i+1}.mp3")