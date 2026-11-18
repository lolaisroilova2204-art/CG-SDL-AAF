# CG-SDL-AAF — foydalanuvchi qo'llanmasi (o'zbekcha)

## 1. O'rnatish
1. **Python 3.10–3.12** ni python.org saytidan o'rnating; o'rnatishda «Add python.exe to PATH» belgisini qo'ying.
2. **Blender 3.6+** ni blender.org dan o'rnating (faqat `.blend` / `.fbx` fayllarni baholash uchun kerak). Dastur uni o'zi topadi; topilmasa «Eksport» oynasida manzilini ko'rsating (masalan `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`).
3. Dastur papkasida buyruq oynasi (cmd) ni ochib:
   ```
   pip install -e .[lpips]
   cgaaf-gui
   ```
4. Kompyuterlarga o'rnatish uchun yagona dastur kerak bo'lsa: `build_windows.bat` ni ishga tushiring — `dist\CG-SDL-AAF\CG-SDL-AAF.exe` hosil bo'ladi.

## 2. Kurs papkasi
Barcha ma'lumotlar bitta papkada saqlanadi (uni Google Drive / tarmoq papkasida ham saqlash mumkin):
```
kurs/
  course.json, students.csv, experts.csv
  tasks/<topshiriq_ID>/           etalon fayllar
  submissions/<talaba_ID>/<topshiriq_ID>/<urinish>/   talaba fayllari + natijalar
  results/                        hisobotlar, rasmlar, dataset
```

## 3. O'qituvchi
1. **Yangi kurs yaratish** → papka va nom.
2. **Topshiriqlar → Topshiriq qo'shish**: ID (masalan `T1_render`), nomlar (uz/en), tur:
   * *Render tasviri* — etalon PNG/JPG yoki `.blend`;
   * *3D model* — etalon OBJ/STL/PLY yoki `.blend`;
   * *Tasvir + 3D model* — eng yaxshisi etalon `.blend`.
   **Blender render rejimi**: «Standart kamera» — etalon va talaba fayllari bir xil kamera, yoritish sozlamasi bilan render qilinadi (tavsiya etiladi).
3. **Guruhni baholash**: Telegram/e-mail orqali kelgan ishlarni `submissions/S001/T1_render/1/` kabi papkalarga joylang va «Baholanmaganlarni baholash» tugmasini bosing.
4. **Ekspert baholari**: «Ko'r baholash oynasi»da avtomatik ball ko'rinmaydi — ikkala ekspert 0–100 ball qo'yadi. Yoki «Ekspert shabloni (Excel)» → to'ldirish → «Import».
5. **Validatsiya va hisobot**: topshiriqni tanlab «Validatsiyani hisoblash». `results/validation_.../` papkasida:
   `report_uz.md`, `report_en.md` (maqola uchun tayyor inglizcha natijalar paragrafi), `tables.xlsx`, `fig_*.png/pdf` (300 dpi).
6. **Kalibrlash**: kamida 8 ta ekspert bahosi bo'lgan topshiriq uchun — ball modeli ekspert baholariga moslashtiriladi va keyingi barcha urinishlarda qo'llanadi.
7. **Eksport**: `dataset.csv` / `dataset.xlsx` — SPSS, R, Python uchun (anonim kodlar bilan).

## 4. Talaba (mustaqil ish)
1. Ro'yxatdan o'zingizni tanlang (yoki «Yangi talaba»).
2. **Rejalashtirish**: maqsad, rejalashtirilgan vaqt, kutilayotgan ball.
3. **Bajarish**: faylni tanlang (`.blend` tavsiya; ekran surati ham qabul qilinadi) → «Tekshirish va topshirish».
4. Natija: ball, baho, metrikalar, farqlar xaritasi, avtomatik izoh va o'z-o'zini baholash aniqligi.
5. **Refleksiya**: nima yaxshi chiqdi, nima qiyin bo'ldi, keyingi safar nimani o'zgartirasiz — «Saqlash».

## 5. Maqola uchun (benchmark + arxiv ma'lumotlari)
«Maqola (benchmark + arxiv)» oynasida uch qadam:
1. **Arxivdagi ishlar.** «Arxiv papkasidan jadval yaratish» → saqlangan barcha fayllar (PNG/JPG, OBJ/STL, `.blend`, `.fbx`) ro'yxati Excel'ga chiqadi. Har bir qatorga `student_id`, `task_id`, `attempt`, `expert1`, `expert2` ni yozing. Bir urinishga tegishli bir nechta fayl (masalan render + model) bir xil `student_id/task_id/attempt` bilan yoziladi. `.max` fayllar «CONVERT» deb belgilanadi — 3ds Max'da FBX ga eksport qiling va jadvaldagi manzilni almashtiring.
   Ekspert baholari qaysi shkalada qo'yilgan bo'lsa (5, 10, 100 …), «shkala maksimumi»ni ko'rsating — dastur 0–100 ga o'tkazadi. Avval «Topshiriqlar» oynasida har bir topshiriq va etalonni kiriting.
2. **Benchmark.** «Benchmarkni ishga tushirish» — talabalarsiz texnik validatsiya: tasvir va modellarga ataylab xato kiritiladi, dasturning sezgirligi, invariantligi, Blender qayta renderining barqarorligi tekshiriladi (1–5 daqiqa).
3. **Maqola natijalari.** «Maqola natijalari to'plamini yaratish» → `paper_results_en.md` (I–VII jadvallar va inglizcha natijalar paragraflari), `paper_tables.xlsx`, `figures/` (300 dpi PNG + PDF).

## 6. Buyruq qatori
`cgaaf --help` — barcha buyruqlar (kurs, topshiriq, baholash, validatsiya, kalibrlash, eksport). `--lang uz|en`.

## 7. Topshiriq sozlamalari (1.2 versiyada qo'shilgan)
* **Baholanadigan obyektlar**: «Avto» — pol, fon tekisliklari kabi yordamchi obyektlar chiqarib tashlanadi; yoki nomlar (`Torus*, Stul*`) yoki kolleksiya bo'yicha.
* **Animatsiya kadrlari soni**: animatsiya topshirig'i uchun 5 (yoki 3–9). Etalon va talaba `.blend` fayli shuncha kadrda render qilinib, har bir kadr solishtiriladi.
* **Ekran suratidan interfeysni avtomatik kesish**: Blender oynasi tushgan ekran suratlaridan 3D ko'rinish maydoni ajratib olinadi. Natijani «Guruhni baholash» oynasida tekshiring: kesilgan ishlar «✂» bilan belgilanadi. Noto'g'ri bo'lsa — «Qo'lda kesish…».
* **O'lchamni tekshirish**: topshiriqda aniq o'lcham talab qilinsa (masalan R = 1) belgilang.
* **Arxiv importida ekspert shkalasi**: 2–5 baholar uchun «5 ballik baho» ni tanlang (5 → 93, 4 → 78, 3 → 62.5, 2 → 27.5), boshqa shkalalar uchun minimum va maksimumni kiriting.
* Validatsiyada har bir talaba–topshiriq juftligidan bitta (oxirgi baholangan) urinish olinadi; ishonch oraliqlari talabalar bo'yicha klasterli bootstrap bilan hisoblanadi.

## 7a. 1.4 versiyadagi o'zgarishlar
* **Tasvir rejimi** (topshiriq sozlamasi): «Obyekt rejimi» (standart) — obyektning kadrdagi joyi va o'lchami baholanmaydi (modellashtirish va material topshiriqlari uchun); belgi olib tashlansa — «kompozitsiya rejimi»: kadr, kamera burilishi va joylashuv ham baholanadi.
* **Xavfsizlik**: talabaning `.blend` fayli ichidagi Python skriptlari hech qachon ishga tushirilmaydi (`--disable-autoexec`).
* **Yordamchi obyektlar**: pol va fon endi poligonlar soniga qaramay (bo'lingan pol, qalin taxta ham) aniqlanadi, modelning kichik yupqa qismlari (plastinka, yozuv taxtasi) esa saqlanadi.
* **Izohlar**: kontur umuman topilmasa — «kontur yetishmaydi» deyiladi; ball 86 dan past bo'lib, birorta aniq qoida ishlamasa ham, eng katta farq bor soha yoki umumiy shakl haqida izoh beriladi. «Ortiqcha qism» izohi hali ishonchsiz — buni talabalarga ayting.
* **Tavsiya**: baholash uchun ekran surati emas, `.blend` faylni talab qiling — benchmarkda ekran suratlari bo'yicha to'g'ri va xato sahnalar deyarli farqlanmadi, qayta render esa barcha to'g'ri sahnalarga 5 baho berdi. Och rangli (light) interfeys mavzusidagi ekran suratlarini avtomatik kesish ishonchsiz — qo'lda kesing.

## 8. Texnik validatsiya maqolasi uchun (talabalar ma'lumotisiz)
1. `pip install -e .[lpips]` va Blender 3.6+ o'rnatilgan bo'lsin.
2. `scripts\run_full_benchmark.bat` ni ishga tushiring (to'liq benchmark Blender renderlari tufayli ancha vaqt oladi).
3. `paper_tech\paper_results_en.md` — maqolaning natijalar bo'limi uchun barcha jadvallar va inglizcha paragraf; `paper_tech\figures\` — 300 dpi rasmlar. Maqoladagi 2–7-rasmlar va jadval qatorlari `scripts/paper/make_figures.py` va `scripts/paper/manuscript_numbers.py` bilan qayta olinadi.
Benchmark haqiqiy talaba ishi emas, boshqariladigan sinov: unda ballning xatoni sezishi, halaqit beruvchi omillarga chidamliligi va oddiy metrikalar (PSNR, Hausdorff va boshqalar) bilan solishtirilganda qanday natija berishi tekshiriladi. 1.4 versiyadagi natijalar `docs/benchmark_reference_run/` papkasida.

## 9. Muhim
* «Namuna (sintetik) kurs» — faqat dasturni sinash uchun; uning raqamlari **maqolada ishlatilmaydi**.
* 3ds Max fayllari (`.max`) to'g'ridan-to'g'ri o'qilmaydi — 3ds Max'da FBX yoki OBJ ga eksport qiling.
* Har bir natijada algoritm versiyasi saqlanadi; `result.json` ni istalgan vaqtda qayta hisoblash mumkin («Hammasini qayta baholash»).
