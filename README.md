# Nested SDE Agricultural Price Path Analysis

**Tarımsal Ürün Fiyat Oluşumu: Stokastik Diferansiyel Denklem Sistemi**

Bu proje, tarımsal ürünlerin tarladan sofraya fiyat oluşumunu nested (iç içe) stokastik diferansiyel denklem (SDE) sistemi ile modelleyen kapsamlı bir analizdir.

## 🎯 Proje Özeti

### Ana Özellikler
- **7 bağımsız SDE modeli** ile tam bağlaşık sistem simülasyonu
- **15,000 Monte Carlo yolu** ile 3 yıllık fiyat patikası analizi
- **Constitutional Mathematical Fidelity** - LaTeX ifadelerinin tam korunması
- **Complex Validation** sistemi ile Pydantic v2 doğrulama
- **Real-time nested integration** - her değişken diğerlerini etkiler

### Sistem Bileşenleri
1. **Üretim Maliyetleri** (C_P) - Geometric Brownian Motion
2. **Lojistik Maliyetleri** (C_L) - Geometric Brownian Motion
3. **Fire Oranı** (ω) - Ornstein-Uhlenbeck süreci
4. **Enflasyon Beklentileri** (Π^E) - Adaptif beklentiler
5. **Üretici Fiyatları** (P_F) - Jump-diffusion süreci
6. **Toptan Fiyatlar** (P_W) - Fire çarpanı etkili
7. **Perakende Fiyatlar** (P_R) - Beklenti-duyarlı marj

## 🚀 Hızlı Başlangıç

### Gereksinimler
```bash
pip install numpy matplotlib scipy pandas sympy pydantic
```

### Temel Kullanım
```bash
python mathematical_analysis.py
```

### VSCode Entegrasyonu
1. `.vscode/launch.json` konfigürasyonu mevcuttur
2. F5 ile debug modunda çalıştırabilirsiniz
3. Tüm sonuçlar Python değişkenlerinde erişilebilir

## 📊 Çıktılar ve Analizler

### Görselleştirmeler
- **16 comprehensive visualization** otomatik oluşturulur
- **Agricultural price path analysis** özel analizi
- Tüm grafikler `analysis_results/` klasörüne kaydedilir

### İstatistiksel Sonuçlar
```
=== TEMEL BULGULAR ===
• Fire Çarpanı: 1.053x
• Toplam Zincir Çarpanı: 2.049x
• Tarla-Tüketici Oranı: 3.093x
• Enflasyon Beklenti Değişimi: +8.74pp
• Volatilite Amplifikasyonu: 0.507x
```

## 🔬 Matematiksel Model

### Nested SDE Sistemi

#### 1. Maliyet Süreçleri
```latex
dC_{P,t} = μ_P C_{P,t} dt + σ_P C_{P,t} dW_{P,t}
dC_{L,t} = μ_L C_{L,t} dt + σ_L C_{L,t} dW_{L,t}
```

#### 2. Fire Oranı Dinamikleri
```latex
dω_t = κ_ω(ω̄ - ω_t)dt + σ_ω dW_{ω,t}
```

#### 3. Enflasyon Beklentileri
```latex
dΠ^E_t = κ_π(π_{R,t} - Π^E_t)dt + σ_π dW_{π,t}
```

#### 4. Üretici Fiyat (Jump-Diffusion)
```latex
dP_{F,t} = κ_F((1+μ_F)C_{P,t} - P_{F,t})dt + σ_F P_{F,t}dW_{F,t} + J_F P_{F,t-}dN_t
```

#### 5. Fire Çarpanı ile Etkin Maliyet
```latex
C^{eff}_{W,t} = \frac{P_{F,t}+C_{L,t}}{1-ω_t}
```

#### 6. Toptan Fiyat
```latex
dP_{W,t} = κ_W((1+μ_W)C^{eff}_{W,t} - P_{W,t})dt + σ_W P_{W,t}dW_{W,t}
```

#### 7. Perakende Fiyat
```latex
dP_{R,t} = κ_R((1+μ_R(Π^E_t))P_{W,t} - P_{R,t})dt + σ_R P_{R,t}dW_{R,t}
```

## ⚙️ Kalibrasyon Parametreleri

### Başlangıç Değerleri
```python
C_P(0) = 20.0    # Üretim maliyeti
C_L(0) = 15.0    # Lojistik maliyeti
ω(0) = 0.05      # Fire oranı (5%)
Π^E(0) = 0.06    # Enflasyon beklentisi (6%)
P_F(0) = 50.0    # Tarla çıkış fiyatı
P_W(0) = 100.0   # Toptan fiyat
P_R(0) = 150.0   # Perakende fiyat
```

### Anahtar Parametreler
| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| κ_F | 3.0 | Üretici fiyat ayarlama hızı |
| κ_W | 2.5 | Toptan fiyat ayarlama hızı |
| κ_R | 1.8 | Perakende fiyat ayarlama hızı |
| μ_F | 0.25 | Üretici kâr marjı (25%) |
| μ_W | 0.20 | Toptan kâr marjı (20%) |
| μ̄_R | 0.30 | Temel perakende marjı (30%) |
| β | 0.5 | Enflasyon duyarlılığı |

## 📁 Proje Yapısı

```
bugun/
├── mathematical_analysis.py          # Ana analiz dosyası
├── simple_mathematical_analysis.py   # Basit SDE versiyonu
├── src/                              # Framework kaynak kodu
│   ├── extraction/                   # Model extraction bileşenleri
│   │   ├── orchestrator.py          # Ana orkestratör
│   │   ├── validators/               # Validation bileşenleri
│   │   │   ├── dependency_analyzer.py
│   │   │   └── sde_classifier.py
│   │   └── models/                   # Pydantic modelleri
├── analysis_results/                 # Otomatik oluşturulan sonuçlar
│   ├── visualization_*.png          # Görselleştirmeler
│   └── analysis_report.txt          # Analiz raporu
├── .vscode/
│   └── launch.json                  # VSCode debug konfigürasyonu
└── README.md                        # Bu dosya
```

## 🔧 Teknik Detaylar

### Complex Validation Sistemi
- **Pydantic v2** ile tam model doğrulama
- **Constitutional Mathematical Fidelity** - LaTeX ifade korunması
- **Enum conversion** - VariableRole → VariableType mapping
- **SDE dimensional validation** - Skaler model kısıtlaması

### Performans
- **15,000 simülasyon yolu** paralel işleme
- **252 zaman adımı** (günlük resolution)
- **3 yıllık** simülasyon penceresi
- **~2 dakika** toplam çalışma süresi

### Unicode Uyumluluğu
- Windows console için ASCII karakter dönüşümü
- Türkçe karakter desteği (ı→i, ş→s, ç→c, etc.)
- [TAG] formatında emoji replacement

## 📈 Analiz Sonuçları

### Fiyat Çarpanları
- **Fire Amplifier**: 1/(1-ω) ≈ 1.053x
- **Supply Chain Multiplier**: Farm→Retail ≈ 3.093x
- **Total System Amplifier**: ≈ 2.049x

### Volatilite Hiyerarşisi
- **Tarla Çıkış**: 43.01% (en yüksek)
- **Toptan**: 26.83% (orta)
- **Perakende**: 21.82% (en düşük)

### Risk Metrikleri
- **VaR (95%)**: 153.88
- **Sharpe Ratio**: 1.44
- **Max Drawdown**: -26.31%

## 🛠️ Geliştirme

### Hata Ayıklama
- Tüm Unicode encoding hataları düzeltildi
- Pydantic validation hataları çözüldü
- SDE dimensional validation eklendi

### Test Edilmiş Ortamlar
- ✅ Windows 10/11 + Python 3.13
- ✅ VSCode + Jupyter extension
- ✅ Command line execution

## 📚 Referanslar

### Akademik Temel
- Stochastic Differential Equations (Øksendal)
- Jump-Diffusion Processes in Finance
- Agricultural Price Formation Models
- Supply Chain Risk Amplification

### Kalibrasyon Kaynakları
- Türkiye gıda piyasası verileri
- TCMB enflasyon beklentileri anketi
- Tarımsal istatistikler (TÜİK)

## 📄 Lisans

Bu proje akademik araştırma amaçlı geliştirilmiştir.

---

**Not**: Analiz sonuçları simülasyon tabanlıdır ve yatırım tavsiyesi niteliği taşımaz.