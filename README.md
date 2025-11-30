

# **NSU AI — Semantic Similarity System**

### **Self-Hosted Embedding Model + Streamlit UI + Prometheus + Grafana + AlertManager + cAdvisor + Node Exporter**

#### **Made by Ahsan Shakoor • 2025**

---

##  **Project Overview**

This project implements a **full MLOps-style semantic similarity system**, including:

### ✔ **Self-hosted LaBSE Model**

* Fully offline, no HuggingFace download
* Saved locally under `models/labse_ahsan_baseline/`
* Encodes two sentences and computes:

  * Cosine Similarity
  * BERTScore F1 (optional)

### ✔ **Modern Streamlit Application**

* Dark mode + neon UI
* Responsive interface
* Computes:

  * Sentence embeddings
  * Cosine similarity
  * BERTScore F1 (optional)
* Exposes Prometheus metrics on **port 8000**

### ✔ **Complete Monitoring Stack**

| Component         | Purpose             | URL                                                            |
| ----------------- | ------------------- | -------------------------------------------------------------- |
| **Prometheus**    | Metrics collection  | [http://localhost:9090](http://localhost:9090)                 |
| **Grafana**       | Dashboards          | [http://localhost:3000](http://localhost:3000)                 |
| **Alertmanager**  | Alerts → Telegram   | [http://localhost:9093](http://localhost:9093)                 |
| **cAdvisor**      | Container metrics   | [http://localhost:8080](http://localhost:8080)                 |
| **Node Exporter** | System metrics      | [http://localhost:9100](http://localhost:9100)                 |
| **Streamlit App** | Your UI             | [http://localhost:8501](http://localhost:8501)                 |
| **App Metrics**   | `/metrics` endpoint | [http://localhost:8000/metrics](http://localhost:8000/metrics) |

### ✔ **Telegram Alerts**

Triggered when:

* App goes DOWN
* High model latency
* Target missing in Prometheus

---
#  **Project Structure**

```
neural-network-experiments/
│
├── app/
│   ├── app.py                   # Streamlit UI + Metrics
│   ├── Dockerfile               # Streamlit service build
│   ├── requirements.txt
│   └── docker-compose.yml       # (if running without monitoring)
│
├── monitoring/
│   ├── docker-compose.yml       # Full monitoring stack
│   ├── prometheus.yml           # Prometheus scrape config
│   ├── alert_rules.yml          # Alerts
│   ├── grafana/                 # Dashboards, provisioning
│   ├── loki/ (optional)
│   └── ...
│
├── models/
│   └── labse_ahsan_baseline/    # Self-hosted model
│
├── experiments/
│   ├── labse_bertscore_baseline.py
│   └── split_and_stats.py
│
├── tapaco_paraphrases_dataset.csv
├── train.csv / val.csv / test.csv
└── README.md
```

---

# **1. How to Run the Full System (App + Monitoring)**

### Inside the `monitoring/` folder:

```bash
cd monitoring
docker compose up -d --build
```

This launches **ALL services**:

* Streamlit UI
* Prometheus
* Grafana
* Alertmanager
* Node Exporter
* cAdvisor

---

# **2. Access the Interface**

### ✔ Streamlit App

 [http://localhost:8501](http://localhost:8501)

### ✔ Prometheus

[http://localhost:9090](http://localhost:9090)

### ✔ Prometheus Targets

[http://localhost:9090/targets](http://localhost:9090/targets)

### ✔ Grafana

[http://localhost:3000](http://localhost:3000)
**Login:**

* user: `admin`
* pass: `admin`

### ✔ cAdvisor

[http://localhost:8080](http://localhost:8080)

### ✔ App Metrics (Prometheus)

 [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

# **3. Prometheus Metrics Exposed by the App**

The app exports these metrics:

```
similarity_requests_total
similarity_latency_seconds_bucket
similarity_latency_seconds_count
similarity_latency_seconds_sum
```

These track:

* Request count
* Latency
* Performance of model inference

---

#  **4. Telegram Alerting Setup**

### Inside `alert_rules.yml`

```yaml
groups:
  - name: app-alerts
    rules:
      - alert: HighLatency
        expr: similarity_latency_seconds_sum / similarity_latency_seconds_count > 3
        for: 20s
        labels:
          severity: critical
        annotations:
          summary: "Similarity service latency is high!"

      - alert: AppDown
        expr: up{job="nsu_similarity_app"} == 0
        for: 15s
        annotations:
          summary: "NSU Similarity App is DOWN!"
```

### AlertManager Config

Inside `alertmanager.yml`:

```yaml
route:
  receiver: telegram

receivers:
  - name: telegram
    telegram_configs:
      - bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
        chat_id: YOUR_CHAT_ID
        message: "⚠ NSU AI Alert:\n{{ .CommonAnnotations.summary }}"
```

Get your chat ID:

```
https://api.telegram.org/bot{TOKEN}/getUpdates
```

---

# **5. Grafana Dashboards**

### Add Prometheus datasource

* Open Grafana → Settings → Data Sources
* Add:

```
http://prometheus:9090
```

### Recommended dashboards:

| Dashboard              | ID                       |
| ---------------------- | ------------------------ |
| Node Exporter Full     | **1860**                 |
| Docker cAdvisor        | **893**                  |
| Python Web App Metrics | custom (we can generate) |

---

# 🧪 **6. Testing alerts**

### Stop the app:

```bash
docker stop nsu_similarity_app
```

Within 20–30 seconds you should get a message:

```
⚠ NSU AI Alert:
NSU Similarity App is DOWN!
```

---

#  **7. Shutdown the stack**

```bash
docker compose down
```

To remove volumes:

```bash
docker compose down -v
```

---

#  **8. Notes & Best Practices**

### ✔ Always monitor latency

Embedding models are heavy — latency alerts help detect overload.

### ✔ Keep model inside image

Ensures repeatable builds.

### ✔ Use alerts that are actionable

Avoid noise.

### ✔ Monitor Prometheus itself

If Prometheus fails → no monitoring.

