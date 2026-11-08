from moviepy import VideoFileClip

def konversi_video_ke_audio(file_video, file_audio):
    try:
        print("Sedang memproses video...")
        # Memuat file video
        video_clip = VideoFileClip(file_video)
        
        # Mengekstrak audio dari video
        audio_clip = video_clip.audio
        
        # Menyimpan audio ke format MP3
        audio_clip.write_audiofile(file_audio)
        
        # Menutup file untuk membebaskan memori
        audio_clip.close()
        video_clip.close()
        
        print(f"Berhasil! Audio disimpan sebagai: {file_audio}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Contoh penggunaan
# Pastikan file video berada di folder yang sama atau tulis path lengkapnya
for i in range(100):
    konversi_video_ke_audio(f"ba/brong_{i+23}.mp4", f"after-knalpot_brong/brong_{i+23}.mp3")
# konversi_video_ke_audio(f"123.mp4", f"123.mp3")