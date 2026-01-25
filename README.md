# 📈 Build and Use Relative Rotation Graphs (RRG) for Smarter Investing using Python

This project demonstrates how to build a **Relative Rotation Graph (RRG)** to analyze sector rotation and relative strength using **Python** and **Streamlit**.  
It helps investors visualize how sectors or stocks are performing relative to a benchmark (like Nifty50 or Nifty Realty).

streamlit run RRG.py

-m pip install -r requirements.txt

streamlit run RRG.py

---

## 🧠 What is an RRG?

A **Relative Rotation Graph (RRG)** displays momentum and relative strength between multiple securities and a benchmark index.  
It divides assets into four quadrants — **Leading**, **Weakening**, **Lagging**, and **Improving** — based on their performance trends.

RRGs help you identify when sectors or stocks are gaining or losing momentum relative to the benchmark, providing an edge in **sector rotation** and **swing trading** strategies.

---

## 🚀 Features

- Fetches **live sector or stock data** using `yfinance`
- Calculates **JdK RS-Ratio** and **JdK RS-Momentum**
- Classifies each stock into one of four RRG quadrants
- Displays interactive **Plotly scatter chart**
- Built with **Streamlit** for a clean and interactive dashboard

---

## 🧩 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
streamlit>=1.25.0
yfinance>=0.2.30
pandas>=2.0.3
numpy>=1.25.2
plotly>=5.17.0
scipy>=1.11.2
```

---

## 🧾 Usage

Run the Streamlit app with:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501` to view the dashboard.

---

## ⚙️ How It Works

1. **Fetch Data** – Uses `yfinance` to download historical price data for each sector or stock.
2. **Compute Relative Strength** – Calculates ratios of stock prices relative to the benchmark.
3. **Smooth and Normalize** – Uses `scipy` to smooth momentum data.
4. **Plot RRG** – Visualizes the data points with `plotly`, classifying them into quadrants.

---

## 🧭 Example Quadrant Interpretation

| Quadrant | Meaning |
|-----------|----------|
| **Leading** | Strong relative strength and rising momentum — outperforming the benchmark |
| **Weakening** | Losing momentum but still relatively strong |
| **Lagging** | Underperforming and weak momentum |
| **Improving** | Gaining momentum and approaching leadership |

---

## 📊 Example Visualization

Once launched, the Streamlit app shows an interactive RRG chart where each bubble represents a stock or sector.  
Hover over each point to view its symbol and live data.

---

## 💡 Insights

- Identify **which sectors are rotating into leadership**
- Detect early signs of **momentum shifts**
- Time **entry and exit** in trending sectors

---

## 📚 Reference

Inspired by the article: [Build and Use Relative Rotation Graphs (RRG) for Smarter Investing using Python](https://fabtrader.in/build-and-use-relative-rotation-graphs-rrg-for-smarter-investing-using-python/)

---

## 🧑‍💻 Author

Developed by **Rajesh A** — passionate about **AI, Machine Learning, and Financial Analytics**.

---

## 🪙 License

This project is open-source and available under the **MIT License**.
# Sector_Rotation-Relative_Rotation_Graph
