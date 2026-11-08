from moviepy import AudioFileClip

def ubah_mpeg_ke_mp3(file_mpeg, file_mp3):
    try:
        print(f"Sedang mengonversi {file_mpeg}...")
        # Membaca trek audio dari file MPEG
        audio = AudioFileClip(file_mpeg)
        
        # Menyimpannya menjadi format MP3
        audio.write_audiofile(file_mp3)
        audio.close()
        
        print(f"Berhasil! File tersimpan sebagai: {file_mp3}\n")
    except Exception as e:
        print(f"Gagal mengonversi: {e}")

# Proses looping untuk 100 file MPEG (.mpeg atau .mpg)
for i in range(100):
    # Sesuaikan ekstensi asal (.mpeg atau .mpg) sesuai dengan file Anda
    file_asal = f"baru/brong_{i+94}.mpeg" 
    file_tujuan = f"baru/brong_{i+94}.mp3"
    
    ubah_mpeg_ke_mp3(file_asal, file_tujuan)
