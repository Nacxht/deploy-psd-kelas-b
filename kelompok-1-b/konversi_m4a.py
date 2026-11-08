from moviepy import AudioFileClip

def ubah_m4a_ke_mp3(file_m4a, file_mp3):
    try:
        print(f"Sedang mengonversi {file_m4a}...")
        # Membaca file m4a
        audio = AudioFileClip(file_m4a)
        
        # Menyimpannya menjadi format MP3
        audio.write_audiofile(file_mp3)
        audio.close()
        
        print(f"Berhasil! File tersimpan sebagai: {file_mp3}\n")
    except Exception as e:
        print(f"Gagal mengonversi: {e}")

# Pastikan file sumber memiliki ekstensi .m4a
for i in range(100):
    ubah_m4a_ke_mp3(f"standart-baru/Gili Timur {i+14}.m4a", f"after-knalpot_standart/standart_{i+30}.mp3")
# ubah_m4a_ke_mp3("bifore-knalpot_standart/standart_10.m4a", "after-knalpot_standart/standart_10.mp3")
# ubah_m4a_ke_mp3("bifore-knalpot_brong/brong_21.mp4", "after-knalpot_brong/brong_21.mp3")
# ubah_m4a_ke_mp3("Rekaman Baru.m4a", "Rekaman Baru.mp3")