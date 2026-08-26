/**
 * Walmart M5 Demand Forecasting — Application Logic & Visualizations
 * ==================================================================
 * Initializes all interactive Plotly & Chart.js visualizations, manages tabs,
 * filters, real-time What-If scenario simulations, and exports.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global Plotly Dark Theme Defaults
  const PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: {
      family: "Inter, -apple-system, sans-serif",
      color: "#9ca3af",
      size: 11
    },
    margin: { t: 30, r: 25, l: 45, b: 40 },
    xaxis: {
      gridcolor: "rgba(255, 255, 255, 0.06)",
      zerolinecolor: "rgba(255, 255, 255, 0.1)",
      tickfont: { color: "#9ca3af", size: 10 }
    },
    yaxis: {
      gridcolor: "rgba(255, 255, 255, 0.06)",
      zerolinecolor: "rgba(255, 255, 255, 0.1)",
      tickfont: { color: "#9ca3af", size: 10 }
    },
    legend: {
      orientation: "h",
      y: 1.12,
      x: 0,
      font: { color: "#e5e7eb", size: 11 }
    },
    hoverlabel: {
      bgcolor: "#1a1a20",
      bordercolor: "rgba(197, 164, 138, 0.4)",
      font: { family: "Inter, sans-serif", color: "#ffffff", size: 12 }
    }
  };

  const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };

  // State Management
  const state = {
    currentTab: "forecast",
    selectedModel: "LightGBM",
    showCi: true,
    showAllModels: false,
    selectedStore: "CA_1",
    selectedCategory: "ALL",
    // Simulator parameters
    simDiscount: 0,
    simSnap: 1.0,
    simWeekend: 0,
    simInflation: 0
  };

  // ══════════════════════════════════════════════════════════════════════════
  // Tab Navigation
  // ══════════════════════════════════════════════════════════════════════════
  function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        if (!target) return;

        tabButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        document.querySelectorAll(".tab-pane").forEach(pane => {
          pane.classList.remove("active");
        });

        const activePane = document.getElementById(`tab-${target}`);
        if (activePane) {
          activePane.classList.add("active");
          state.currentTab = target;
          renderCurrentTab(target);
        }
      });
    });
  }

  function renderCurrentTab(tabId) {
    setTimeout(() => {
      if (tabId === "forecast") renderForecastExplorer();
      else if (tabId === "eda") renderEdaCharts();
      else if (tabId === "leaderboard") renderLeaderboard();
      else if (tabId === "insights") renderInsights();
      else if (tabId === "simulator") renderSimulator();
      else if (tabId === "api") renderApiPlayground();
    }, 50);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 1. Forecast Explorer
  // ══════════════════════════════════════════════════════════════════════════
  function renderForecastExplorer() {
    const container = document.getElementById("forecast-chart");
    if (!container) return;

    const traces = [];

    // Trace 1: Ground Truth Actuals
    traces.push({
      x: M5_DATA.testDates,
      y: M5_DATA.actualSales,
      mode: "lines+markers",
      name: "Actual Sales (Holdout)",
      line: { color: "#ffffff", width: 2.5 },
      marker: { size: 5, color: "#ffffff" }
    });

    if (state.showAllModels) {
      // Overlay all 5 models
      M5_DATA.modelMetrics.forEach(m => {
        const forecast = M5_DATA.modelForecasts[m.shortName || m.model];
        if (forecast) {
          traces.push({
            x: M5_DATA.testDates,
            y: forecast.predicted,
            mode: "lines",
            name: `${m.shortName || m.model} (RMSE: ${m.rmse})`,
            line: { color: m.color, width: m.isBest ? 2.5 : 1.8, dash: m.isBest ? "solid" : "dot" }
          });
        }
      });
    } else {
      // Selected Model with Confidence Interval Band
      const activeModelInfo = M5_DATA.modelMetrics.find(m => (m.shortName || m.model) === state.selectedModel) || M5_DATA.modelMetrics[0];
      const forecast = M5_DATA.modelForecasts[state.selectedModel] || M5_DATA.modelForecasts.LightGBM;

      if (state.showCi && forecast.ci_upper && forecast.ci_lower) {
        // CI Upper
        traces.push({
          x: M5_DATA.testDates,
          y: forecast.ci_upper,
          mode: "lines",
          line: { width: 0, color: "transparent" },
          showlegend: false,
          hoverinfo: "skip"
        });
        // CI Lower with fill
        traces.push({
          x: M5_DATA.testDates,
          y: forecast.ci_lower,
          mode: "lines",
          fill: "tonexty",
          fillcolor: activeModelInfo.color === "#10b981" ? "rgba(16, 185, 129, 0.15)" : "rgba(197, 164, 138, 0.18)",
          line: { width: 0, color: "transparent" },
          name: "95% Prediction Interval"
        });
      }

      // Predicted line
      traces.push({
        x: M5_DATA.testDates,
        y: forecast.predicted,
        mode: "lines+markers",
        name: `${state.selectedModel} Forecast`,
        line: { color: activeModelInfo.color, width: 2.8 },
        marker: { size: 6, color: activeModelInfo.color }
      });
    }

    const layout = {
      ...PLOTLY_DARK_LAYOUT,
      title: false,
      xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Date (28-Day Holdout Evaluation Window)" },
      yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Aggregated Daily Units Sold" },
      height: 420
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
    renderResidualsChart();
  }

  function renderResidualsChart() {
    const container = document.getElementById("residuals-chart");
    if (!container) return;

    const forecast = M5_DATA.modelForecasts[state.selectedModel] || M5_DATA.modelForecasts.LightGBM;
    const residuals = M5_DATA.actualSales.map((act, i) => Math.round((act - forecast.predicted[i]) * 10) / 10);

    const trace = {
      x: M5_DATA.testDates,
      y: residuals,
      type: "bar",
      name: "Residual Error (Actual - Predicted)",
      marker: {
        color: residuals.map(r => r >= 0 ? "#10b981" : "#ef4444")
      }
    };

    const layout = {
      ...PLOTLY_DARK_LAYOUT,
      title: false,
      xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Date" },
      yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Residual Error (Units)" },
      height: 240,
      margin: { t: 15, r: 25, l: 45, b: 35 }
    };

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 2. Exploratory Data Analysis (EDA)
  // ══════════════════════════════════════════════════════════════════════════
  function renderEdaCharts() {
    // 5-Year Historical Time Series
    const histContainer = document.getElementById("eda-history-chart");
    if (histContainer) {
      const traces = [
        {
          x: M5_DATA.historicalDates,
          y: M5_DATA.historicalSales,
          mode: "lines",
          name: "Weekly Aggregated Sales",
          line: { color: "#c5a48a", width: 1.6 }
        },
        {
          x: M5_DATA.historicalDates,
          y: M5_DATA.historicalTrend,
          mode: "lines",
          name: "Macro Growth Trend (2011–2016)",
          line: { color: "#10b981", width: 2, dash: "dash" }
        }
      ];

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Timeline (1,941 Days / 5.4 Years)" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Total Units Sold (Weekly Sample)" },
        height: 340
      };

      Plotly.newPlot(histContainer, traces, layout, PLOTLY_CONFIG);
    }

    // Category Donut Chart
    const catContainer = document.getElementById("eda-category-chart");
    if (catContainer) {
      const trace = {
        labels: M5_DATA.categoryData.labels,
        values: M5_DATA.categoryData.shares,
        type: "pie",
        hole: 0.58,
        marker: {
          colors: M5_DATA.categoryData.colors
        },
        textinfo: "label+percent",
        textfont: { color: "#ffffff", size: 12 }
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        showlegend: false,
        height: 320,
        margin: { t: 20, r: 20, l: 20, b: 20 }
      };

      Plotly.newPlot(catContainer, [trace], layout, PLOTLY_CONFIG);
    }

    // Store Performance Bar Chart
    const storeContainer = document.getElementById("eda-store-chart");
    if (storeContainer) {
      const trace = {
        x: M5_DATA.storeData.map(s => s.storeId),
        y: M5_DATA.storeData.map(s => s.units),
        type: "bar",
        marker: {
          color: M5_DATA.storeData.map(s => s.state === "California" ? "#c5a48a" : (s.state === "Texas" ? "#3b82f6" : "#f59e0b"))
        },
        text: M5_DATA.storeData.map(s => s.growth),
        textposition: "auto",
        textfont: { color: "#ffffff", size: 10 }
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Walmart Store ID (CA, TX, WI)" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Total Volume (Units)" },
        height: 320
      };

      Plotly.newPlot(storeContainer, [trace], layout, PLOTLY_CONFIG);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 3. Model Benchmark Leaderboard
  // ══════════════════════════════════════════════════════════════════════════
  function renderLeaderboard() {
    const tableBody = document.getElementById("leaderboard-tbody");
    if (tableBody) {
      tableBody.innerHTML = M5_DATA.modelMetrics.map((m, idx) => `
        <tr class="${m.isBest ? 'winner-row' : ''}">
          <td style="font-weight: 700; color: #ffffff;">
            #${idx + 1}
          </td>
          <td>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span style="width: 10px; height: 10px; border-radius: 50%; background: ${m.color}; display: inline-block;"></span>
              <strong>${m.model}</strong>
              ${m.isBest ? '<span class="winner-badge">Champion</span>' : ''}
            </div>
            <div style="font-size: 0.74rem; color: var(--text-muted);">${m.type}</div>
          </td>
          <td><span class="code-pill">${m.rmse.toFixed(2)}</span></td>
          <td>${m.mae.toFixed(2)}</td>
          <td>${m.mape.toFixed(1)}%</td>
          <td>${m.smape.toFixed(1)}%</td>
          <td><strong style="color: ${m.isBest ? 'var(--accent-green)' : 'var(--text-primary)'};">${m.wrmsse.toFixed(3)}</strong></td>
          <td>${m.trainTime.toFixed(1)}s</td>
          <td><span style="color: var(--accent-sand); font-size: 0.78rem;">${m.inferenceLatency}</span></td>
        </tr>
      `).join("");
    }

    // Benchmark Radar Chart
    const radarContainer = document.getElementById("leaderboard-radar-chart");
    if (radarContainer) {
      const categories = ["Accuracy (1/RMSE)", "Precision (1/MAE)", "Percentage (1/MAPE)", "Competition (1/WRMSSE)", "Train Speed"];
      
      const traces = M5_DATA.modelMetrics.map(m => {
        // Normalize metrics to 0-100 scale where higher is better
        const normRmse = Math.max(10, 100 - (m.rmse - 1.84) * 50);
        const normMae = Math.max(10, 100 - (m.mae - 1.21) * 60);
        const normMape = Math.max(10, 100 - (m.mape - 10.8) * 8);
        const normWrmsse = Math.max(10, 100 - (m.wrmsse - 0.493) * 180);
        const normSpeed = Math.max(10, 100 - (m.trainTime / 312.7) * 90);

        return {
          type: "scatterpolar",
          r: [normRmse, normMae, normMape, normWrmsse, normSpeed, normRmse],
          theta: [...categories, categories[0]],
          fill: "toself",
          fillcolor: m.isBest ? "rgba(16, 185, 129, 0.2)" : "rgba(255, 255, 255, 0.03)",
          name: m.shortName || m.model,
          line: { color: m.color, width: m.isBest ? 2.5 : 1.5 }
        };
      });

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        polar: {
          radialaxis: {
            visible: true,
            range: [0, 100],
            color: "rgba(255, 255, 255, 0.2)",
            tickfont: { color: "#6b7280", size: 8 }
          },
          angularaxis: {
            color: "rgba(255, 255, 255, 0.2)",
            tickfont: { color: "#e5e7eb", size: 10 }
          },
          bgcolor: "transparent"
        },
        height: 380,
        margin: { t: 30, r: 40, l: 40, b: 30 }
      };

      Plotly.newPlot(radarContainer, traces, layout, PLOTLY_CONFIG);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 4. Model Insights & Interpretability
  // ══════════════════════════════════════════════════════════════════════════
  function renderInsights() {
    // Feature Importance Chart
    const fiContainer = document.getElementById("insights-fi-chart");
    if (fiContainer) {
      const trace = {
        y: M5_DATA.lgbFeatureImportance.map(f => f.feature).reverse(),
        x: M5_DATA.lgbFeatureImportance.map(f => f.importance).reverse(),
        type: "bar",
        orientation: "h",
        marker: {
          color: "rgba(197, 164, 138, 0.85)",
          line: { color: "#c5a48a", width: 1 }
        }
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "LightGBM Total Split Gain Importance" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, automargin: true },
        height: 380,
        margin: { t: 20, r: 25, l: 180, b: 40 }
      };

      Plotly.newPlot(fiContainer, [trace], layout, PLOTLY_CONFIG);
    }

    // TFT Attention Weights Chart
    const attnContainer = document.getElementById("insights-tft-chart");
    if (attnContainer) {
      const trace = {
        x: M5_DATA.tftAttention.map(a => `Day +${a.horizon}`),
        y: M5_DATA.tftAttention.map(a => a.weight),
        type: "bar",
        marker: {
          color: "#a855f7"
        },
        text: M5_DATA.tftAttention.map(a => a.focus),
        textposition: "auto"
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Multi-Horizon Forecast Step" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Self-Attention Weight" },
        height: 320
      };

      Plotly.newPlot(attnContainer, [trace], layout, PLOTLY_CONFIG);
    }

    // Prophet Weekly Seasonality Curve
    const prophetContainer = document.getElementById("insights-prophet-chart");
    if (prophetContainer) {
      const trace = {
        x: M5_DATA.prophetComponents.weeklySeasonality.map(w => w.day),
        y: M5_DATA.prophetComponents.weeklySeasonality.map(w => w.effect),
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#3b82f6", width: 2.5 },
        marker: { size: 7, color: "#3b82f6" },
        fill: "tozeroy",
        fillcolor: "rgba(59, 130, 246, 0.12)"
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Day of Week" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Additive Unit Shift Effect" },
        height: 320
      };

      Plotly.newPlot(prophetContainer, [trace], layout, PLOTLY_CONFIG);
    }

    // SARIMA Autocorrelation (ACF)
    const sarimaContainer = document.getElementById("insights-sarima-chart");
    if (sarimaContainer) {
      const trace = {
        x: M5_DATA.sarimaDiagnostics.lags,
        y: M5_DATA.sarimaDiagnostics.acf,
        type: "bar",
        marker: {
          color: M5_DATA.sarimaDiagnostics.acf.map((v, i) => (i % 7 === 0 && i > 0) ? "#10b981" : "#8c6c53")
        }
      };

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Lag (Days — Note 7-day Weekly Peaks)" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Autocorrelation (ACF)" },
        height: 320
      };

      Plotly.newPlot(sarimaContainer, [trace], layout, PLOTLY_CONFIG);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 5. Interactive What-If Scenario Simulator
  // ══════════════════════════════════════════════════════════════════════════
  function renderSimulator() {
    const baseForecast = M5_DATA.modelForecasts[state.selectedModel] || M5_DATA.modelForecasts.LightGBM;
    const avgPrice = 3.65; // Walmart M5 typical basket unit price ($)

    // Calculate elasticity and scenario adjustments
    // Discount: Price elasticity ~ -1.45 (10% discount -> +14.5% unit sales)
    const discountMultiplier = 1.0 + (Math.abs(state.simDiscount) / 100.0) * 1.45;
    // SNAP effect: boosts sales on days 1-10 of month
    const snapMultiplier = state.simSnap;
    // Weekend boost: adds to Saturday/Sunday (indices 5, 6, 12, 13, etc.)
    const weekendMultiplier = 1.0 + (state.simWeekend / 100.0);
    // Inflation drag: -0.6 elasticity
    const inflationMultiplier = 1.0 - (state.simInflation / 100.0) * 0.6;

    const simulatedSales = baseForecast.predicted.map((baseVal, idx) => {
      const date = new Date(M5_DATA.testDates[idx]);
      const dayOfWeek = date.getDay(); // 0 is Sunday, 6 is Saturday
      const dayOfMonth = date.getDate();

      let val = baseVal * discountMultiplier * inflationMultiplier;
      if (dayOfMonth <= 10) val *= snapMultiplier;
      if (dayOfWeek === 0 || dayOfWeek === 6) val *= weekendMultiplier;

      return Math.round(val * 10) / 10;
    });

    // Summary calculations
    const baseTotalUnits = baseForecast.predicted.reduce((a, b) => a + b, 0);
    const simTotalUnits = simulatedSales.reduce((a, b) => a + b, 0);
    const unitChangePct = ((simTotalUnits - baseTotalUnits) / baseTotalUnits) * 100.0;

    const baseRevenue = baseTotalUnits * avgPrice;
    const simPrice = avgPrice * (1.0 - Math.abs(state.simDiscount) / 100.0);
    const simRevenue = simTotalUnits * simPrice;
    const revChangePct = ((simRevenue - baseRevenue) / baseRevenue) * 100.0;

    // Update UI Impact Box
    const impactValEl = document.getElementById("sim-revenue-impact");
    const impactUnitsEl = document.getElementById("sim-units-impact");
    if (impactValEl) {
      const prefix = revChangePct >= 0 ? "+" : "";
      impactValEl.textContent = `${prefix}${revChangePct.toFixed(1)}% ($${Math.round(simRevenue).toLocaleString()})`;
      impactValEl.style.color = revChangePct >= 0 ? "var(--accent-green)" : "var(--accent-red)";
    }
    if (impactUnitsEl) {
      const prefix = unitChangePct >= 0 ? "+" : "";
      impactUnitsEl.textContent = `${prefix}${unitChangePct.toFixed(1)}% (${Math.round(simTotalUnits).toLocaleString()} Units)`;
    }

    // Render Scenario Chart
    const simContainer = document.getElementById("simulator-chart");
    if (simContainer) {
      const traces = [
        {
          x: M5_DATA.testDates,
          y: baseForecast.predicted,
          mode: "lines",
          name: `Base Forecast (${state.selectedModel})`,
          line: { color: "#9ca3af", width: 2, dash: "dash" }
        },
        {
          x: M5_DATA.testDates,
          y: simulatedSales,
          mode: "lines+markers",
          name: "Simulated Scenario Demand",
          line: { color: revChangePct >= 0 ? "#10b981" : "#ef4444", width: 2.8 },
          marker: { size: 6, color: revChangePct >= 0 ? "#10b981" : "#ef4444" },
          fill: "tonexty",
          fillcolor: revChangePct >= 0 ? "rgba(16, 185, 129, 0.12)" : "rgba(239, 68, 68, 0.12)"
        }
      ];

      const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: "Date (28-Day Simulation Horizon)" },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: "Forecasted Daily Units" },
        height: 380
      };

      Plotly.newPlot(simContainer, traces, layout, PLOTLY_CONFIG);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // 6. REST API Playground & Export
  // ══════════════════════════════════════════════════════════════════════════
  function renderApiPlayground() {
    const apiPayloadEl = document.getElementById("api-json-output");
    if (apiPayloadEl) {
      const forecast = M5_DATA.modelForecasts[state.selectedModel] || M5_DATA.modelForecasts.LightGBM;
      const payload = {
        status: "success",
        timestamp: new Date().toISOString(),
        metadata: {
          store_id: state.selectedStore,
          item_id: "FOODS_3_090",
          model_paradigm: state.selectedModel,
          horizon_days: 28,
          confidence_interval_level: "95%"
        },
        evaluation_metrics: M5_DATA.modelMetrics.find(m => (m.shortName || m.model) === state.selectedModel),
        daily_forecast: M5_DATA.testDates.map((date, idx) => ({
          date: date,
          predicted_units: forecast.predicted[idx],
          ci_lower_95: forecast.ci_lower[idx],
          ci_upper_95: forecast.ci_upper[idx]
        }))
      };

      apiPayloadEl.textContent = JSON.stringify(payload, null, 2);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // UI Controls & Event Listeners
  // ══════════════════════════════════════════════════════════════════════════
  function initControls() {
    // Model Selector
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
      modelSelect.addEventListener("change", (e) => {
        state.selectedModel = e.target.value;
        if (state.currentTab === "forecast") renderForecastExplorer();
        else if (state.currentTab === "simulator") renderSimulator();
        else if (state.currentTab === "api") renderApiPlayground();
      });
    }

    // CI Toggle
    const ciToggle = document.getElementById("ci-toggle");
    if (ciToggle) {
      ciToggle.addEventListener("click", () => {
        state.showCi = !state.showCi;
        ciToggle.classList.toggle("active", state.showCi);
        renderForecastExplorer();
      });
    }

    // All Models Overlay Toggle
    const overlayToggle = document.getElementById("overlay-toggle");
    if (overlayToggle) {
      overlayToggle.addEventListener("click", () => {
        state.showAllModels = !state.showAllModels;
        overlayToggle.classList.toggle("active", state.showAllModels);
        renderForecastExplorer();
      });
    }

    // Simulator Sliders
    const discountSlider = document.getElementById("sim-discount");
    const snapSlider = document.getElementById("sim-snap");
    const weekendSlider = document.getElementById("sim-weekend");
    const inflationSlider = document.getElementById("sim-inflation");

    if (discountSlider) {
      discountSlider.addEventListener("input", (e) => {
        state.simDiscount = parseFloat(e.target.value);
        document.getElementById("val-discount").textContent = `${state.simDiscount}%`;
        renderSimulator();
      });
    }
    if (snapSlider) {
      snapSlider.addEventListener("input", (e) => {
        state.simSnap = parseFloat(e.target.value);
        document.getElementById("val-snap").textContent = `${state.simSnap.toFixed(2)}x`;
        renderSimulator();
      });
    }
    if (weekendSlider) {
      weekendSlider.addEventListener("input", (e) => {
        state.simWeekend = parseFloat(e.target.value);
        document.getElementById("val-weekend").textContent = `+${state.simWeekend}%`;
        renderSimulator();
      });
    }
    if (inflationSlider) {
      inflationSlider.addEventListener("input", (e) => {
        state.simInflation = parseFloat(e.target.value);
        document.getElementById("val-inflation").textContent = `${state.simInflation > 0 ? '+' : ''}${state.simInflation}%`;
        renderSimulator();
      });
    }

    // Export CSV Button
    const exportCsvBtn = document.getElementById("btn-export-csv");
    if (exportCsvBtn) {
      exportCsvBtn.addEventListener("click", () => {
        const forecast = M5_DATA.modelForecasts[state.selectedModel] || M5_DATA.modelForecasts.LightGBM;
        let csvContent = "data:text/csv;charset=utf-8,date,actual,predicted,ci_lower,ci_upper\n";
        M5_DATA.testDates.forEach((d, i) => {
          csvContent += `${d},${M5_DATA.actualSales[i]},${forecast.predicted[i]},${forecast.ci_lower[i]},${forecast.ci_upper[i]}\n`;
        });
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `m5_${state.selectedModel.toLowerCase()}_forecast.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
    }
  }

  // Initialize Everything
  initTabs();
  initControls();
  renderForecastExplorer();
});
