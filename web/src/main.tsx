import {
  Activity,
  ArrowUpRight,
  ChartNoAxesCombined,
  Download,
  FlaskConical,
  Orbit,
  ScanLine,
  ShieldCheck,
} from "lucide-react";
import type { Data, Layout } from "plotly.js";
import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
// The distribution bundle has the same public plotting API.
// @ts-expect-error Distribution package ships no declarations.
import Plotly from "plotly.js-dist-min";

type Metric = {
  n: number;
  available: boolean;
  accuracy?: number;
  balanced_accuracy?: number;
  macro_f1?: number;
  weighted_f1?: number;
  warning?: string;
  class_order?: string[];
  confusion_matrix?: number[][];
  support: Record<string, number>;
  per_class?: Record<
    string,
    {
      precision: number | null;
      recall: number | null;
      f1: number | null;
      support: number;
    }
  >;
};
type Ablation = {
  index: number;
  group: string;
  metrics: Metric;
  macro_f1_drop: number;
  changed_n: number;
  changed_fraction: number;
  mean_max_score_delta: number;
  by_mission: Record<string, { n: number; changed: number }>;
  by_class: Record<string, { n: number; changed: number }>;
};
type Run = {
  id: string;
  experiment: string;
  dataset_id: string;
  timestamp: string;
  audit_completed_at: string;
  classes: string[];
  models: Record<
    string,
    {
      train_n: number;
      test: Metric;
      validation: Metric;
      missions: Record<string, Metric>;
      features: string[];
    }
  >;
  ablations: Ablation[];
  limitations: string[];
};
type Group = {
  mission: string;
  class: string;
  n: number;
  missing_transit: number;
  zero_folded: number;
  missing_flags: number;
};
type Manifest = {
  dataset_id: string;
  revision: string;
  actual_sizes: Record<string, number>;
  config: { mode: string; seed: number };
  summary: {
    classes: Record<string, number>;
    missions: Record<string, number>;
    groups: Group[];
    artifact_counts: Record<string, number>;
  };
  quality: Record<string, unknown>;
  compressed_flux_row_group_bytes: number;
  inspected_rows: Record<string, number>;
};
type Row = {
  target_id: string;
  star_id: string;
  mission: string;
  disposition: string;
  split: string;
  [key: string]: string | number;
};
type Page = {
  total: number;
  items: Row[];
  metrics?: Record<string, Metric>;
  error_breakdown?: Record<string, number>;
};
type Target = {
  metadata: Record<string, string | number | boolean | null>;
  shortcut_features: Record<string, string | number>;
  signal_features: Record<string, number>;
};
type Flux = {
  x: number[];
  y: (number | null)[];
  unavailable: boolean;
  annotation: string | null;
  x_label: string;
  original_n: number;
};
const amber = "#edb66a",
  cyan = "#70cdd4";
const pct = (v?: number | null) =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;
const nice = (s: string) => s.replaceAll("_", " ");
async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`/api${path}`, { signal });
  if (!r.ok) {
    const e = await r.json();
    throw new Error(e.detail || "Request failed");
  }
  return r.json();
}
function useApi<T>(path: string | null) {
  const [state, set] = useState<{
    value?: T;
    error?: string;
    loading: boolean;
  }>({ loading: true });
  useEffect(() => {
    const c = new AbortController();
    set({ loading: true });
    if (!path) {
      set({ loading: false });
      return;
    }
    api<T>(path, c.signal)
      .then((value) => set({ value, loading: false }))
      .catch((e) => {
        if (e.name !== "AbortError")
          set({ error: String(e.message), loading: false });
      });
    return () => c.abort();
  }, [path]);
  return state;
}
function Chart({
  data,
  title,
  xTitle,
  yTitle,
  height = 300,
}: {
  data: Data[];
  title: string;
  xTitle?: string;
  yTitle?: string;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current!;
    const layout: Partial<Layout> = {
      title: { text: title, font: { size: 13, color: "#adbcc9" } },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#c7d0da", family: "ui-monospace, monospace", size: 11 },
      margin: { l: 55, r: 25, t: 50, b: 65 },
      height,
      autosize: true,
      xaxis: {
        title: { text: xTitle },
        gridcolor: "#23303d",
        automargin: true,
        showspikes: true,
      },
      yaxis: {
        title: { text: yTitle },
        gridcolor: "#23303d",
        zerolinecolor: "#23303d",
        automargin: true,
      },
      legend: { orientation: "h", y: 1.15 },
      barmode: "group",
      hovermode: "closest",
    };
    Plotly.react(el, data, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
    const o = new ResizeObserver(() => Plotly.Plots.resize(el));
    o.observe(el);
    return () => {
      o.disconnect();
      Plotly.purge(el);
    };
  }, [data, title, xTitle, yTitle, height]);
  return <div ref={ref} role="img" aria-label={title} className="chart" />;
}
function Notice({ children }: { children: React.ReactNode }) {
  return <div className="notice">{children}</div>;
}
function App() {
  const [screen, setScreen] = useState("Overview");
  const [selectedRun, setSelectedRun] = useState("");
  const [task, setTask] = useState("multiclass");
  const [target, setTarget] = useState("");
  const [discovery, setDiscovery] = useState("");
  const [refresh, setRefresh] = useState(0);
  const summary = useApi<{ data: Manifest }>(
      `/dataset/summary?refresh=${refresh}`,
    ),
    experiments = useApi<{ data: Run[] }>(`/experiments?refresh=${refresh}`);
  const allRuns = experiments.value?.data ?? [];
  const eligible = allRuns.filter((r) => r.experiment === task);
  const run = eligible.find((r) => r.id === selectedRun) ?? eligible[0];
  const manifest = summary.value?.data;
  const openTarget = (id: string) => {
    setTarget(id);
    setScreen("Signal Explorer");
  };
  return (
    <div className="app">
      <aside>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setScreen("Overview");
          }}
        >
          <Orbit size={28} />
          <span>
            Transit<span className="light">Shortcut</span>
            <small>SCIENTIFIC ML AUDITOR</small>
          </span>
        </a>
        <div className="navlabel">WORKSTATION / 01</div>
        <nav>
          {[
            ["Overview", Activity],
            ["Model Comparison", ChartNoAxesCombined],
            ["Signal Explorer", ScanLine],
            ["Shortcut Audit", ShieldCheck],
          ].map(([label, Icon]) => {
            const I = Icon as typeof Activity;
            return (
              <button
                key={String(label)}
                className={screen === label ? "active" : ""}
                onClick={() => setScreen(String(label))}
              >
                <I size={17} />
                {String(label)}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-note">
          <span className="dot" /> LOCAL CPU WORKFLOW
          <p>
            Correlation ≠ reliance
            <br />
            Reliance ≠ generalization
          </p>
          <small>Kepler · K2 · TESS</small>
        </div>
      </aside>
      <main>
        <header>
          <span>
            OBSERVATORY <span className="muted">/ {screen.toUpperCase()}</span>
          </span>
          <button onClick={() => setRefresh((v) => v + 1)}>
            Refresh artifacts ↻
          </button>
        </header>
        <div className="content">
          <div className="eyebrow">
            EXOPLANET CLASSIFICATION / DATASET INTEGRITY
          </div>
          <div className="heading">
            <div>
              <h1>{screen}</h1>
              <p>
                {screen === "Overview"
                  ? "Is your classifier detecting transits—or learning the dataset?"
                  : screen === "Model Comparison"
                    ? "Two feature families. One evaluation cohort."
                    : screen === "Signal Explorer"
                      ? "Inspect observations, artifacts, and model disagreements."
                      : "Measure suspicious correlations and refit sensitivity."}
              </p>
            </div>
            <span className="badge">
              {manifest
                ? `${manifest.config.mode.toUpperCase()} · SEED ${manifest.config.seed}`
                : "AWAITING DATA"}
            </span>
          </div>
          {summary.loading && <Notice>Loading local dataset artifacts…</Notice>}
          {!summary.loading && !manifest && (
            <section className="setup">
              <FlaskConical size={32} />
              <h2>Begin with an auditable sample.</h2>
              <p>
                No dataset or benchmark results are bundled. Prepare actual
                observations, train both baselines, then refresh this
                workstation.
              </p>
              <pre>
                uv run transitshortcut data prepare --mode sample --seed 42
                <br />
                uv run transitshortcut train --experiment multiclass
                <br />
                uv run transitshortcut train --experiment cross-mission
              </pre>
              <p className="muted">
                {summary.error} The CLI explains the flux download budget before
                exceeding it.
              </p>
              <div className="grid two">
                <div>
                  <h3>Shortcut Model</h3>
                  <p>
                    Metadata availability, mission identity, and deliberately
                    selected preprocessing indicators.
                  </p>
                </div>
                <div>
                  <h3>Signal Model</h3>
                  <p>
                    Interpretable numerical features extracted exclusively from
                    raw photometry.
                  </p>
                </div>
              </div>
            </section>
          )}
          {manifest && (
            <>
              <div className="statusline">
                <span>
                  <i className="dot" />
                  Dataset prepared
                </span>
                <span>
                  {allRuns.length ? "● Models trained" : "○ Models not trained"}
                </span>
                <span>REV {manifest.revision.slice(0, 12)}</span>
                <span>
                  {Object.values(manifest.actual_sizes)
                    .reduce((a, b) => a + b, 0)
                    .toLocaleString()}{" "}
                  selected rows
                </span>
              </div>
              {screen !== "Overview" && (
                <div className="toolbar">
                  <label>
                    Experiment
                    <select
                      aria-label="Experiment"
                      value={task}
                      onChange={(e) => {
                        setTask(e.target.value);
                        setSelectedRun("");
                        setTarget("");
                      }}
                    >
                      <option value="multiclass">
                        A · Three-class classification
                      </option>
                      <option value="cross-mission">
                        B · Binary cross-mission comparison
                      </option>
                    </select>
                  </label>
                  <label>
                    Completed run
                    <select
                      aria-label="Completed run"
                      value={run?.id ?? ""}
                      onChange={(e) => setSelectedRun(e.target.value)}
                    >
                      {!eligible.length && <option>No completed run</option>}
                      {eligible.map((r) => (
                        <option key={r.id}>{r.id}</option>
                      ))}
                    </select>
                  </label>
                  {run && (
                    <a
                      className="button"
                      href={`/api/reports/${run.id}?format=markdown`}
                    >
                      <Download size={15} /> Audit report
                    </a>
                  )}
                </div>
              )}
              {screen === "Overview" && (
                <>
                  <section className="intro">
                    <span className="eyebrow">
                      THE QUESTION BEHIND THE SCORE
                    </span>
                    <h2>
                      High accuracy needs
                      <br />
                      <em>an explanation.</em>
                    </h2>
                    <p>
                      Dataset construction can leave predictive traces. Compare
                      a metadata-only baseline with raw-photometry features,
                      then remove suspicious feature groups and measure what
                      changes.
                    </p>
                    <button
                      className="primary"
                      onClick={() => setScreen("Model Comparison")}
                    >
                      Examine model evidence <ArrowUpRight size={16} />
                    </button>
                  </section>
                  <div className="grid two">
                    <section>
                      <h2>Observed cohort composition</h2>
                      <Chart
                        title="Mission × class · selected rows"
                        data={Object.keys(manifest.summary.classes).map(
                          (c, i) => ({
                            type: "bar",
                            name: nice(c),
                            x: ["kepler", "k2", "tess"],
                            y: ["kepler", "k2", "tess"].map(
                              (m) =>
                                manifest.summary.groups.find(
                                  (g) => g.mission === m && g.class === c,
                                )?.n ?? 0,
                            ),
                            marker: { color: [cyan, amber, "#8190b7"][i] },
                          }),
                        )}
                      />
                    </section>
                    <section>
                      <h2>Construction traces</h2>
                      {Object.entries(manifest.summary.artifact_counts).map(
                        ([k, v]) => (
                          <div className="statrow" key={k}>
                            <span>{nice(k)}</span>
                            <strong>
                              {v.toLocaleString()} <small>rows</small>
                            </strong>
                          </div>
                        ),
                      )}
                      <p className="muted">
                        Observed prevalence establishes association, not model
                        reliance. Refit ablations evaluate sensitivity.
                      </p>
                      <Notice>
                        NO_SIGNAL has documented Kepler-only coverage. Use the
                        separate binary task for comparable mission breakdowns.
                      </Notice>
                    </section>
                  </div>
                  {allRuns.find((r) => r.experiment === "multiclass") && (
                    <section>
                      <h2>Measured result · three-class experiment</h2>
                      <p>
                        On{" "}
                        {
                          allRuns.find((r) => r.experiment === "multiclass")!
                            .models.shortcut.test.n
                        }{" "}
                        test observations, Shortcut Model macro F1 is{" "}
                        <strong className="amber">
                          {pct(
                            allRuns.find((r) => r.experiment === "multiclass")!
                              .models.shortcut.test.macro_f1,
                          )}
                        </strong>{" "}
                        and Signal Model macro F1 is{" "}
                        <strong className="cyan">
                          {pct(
                            allRuns.find((r) => r.experiment === "multiclass")!
                              .models.signal.test.macro_f1,
                          )}
                        </strong>
                        . This is a sampled in-distribution comparison. Feature
                        restrictions and refit sensitivity do not establish
                        physical understanding.
                      </p>
                      <p className="mono muted">
                        Last completed audit:{" "}
                        {
                          allRuns.find((r) => r.experiment === "multiclass")!
                            .audit_completed_at
                        }
                      </p>
                    </section>
                  )}
                  <section>
                    <h2>Experiment log</h2>
                    {allRuns.length ? (
                      allRuns.map((r) => (
                        <button
                          className="runrow"
                          key={r.id}
                          onClick={() => {
                            setTask(r.experiment);
                            setSelectedRun(r.id);
                            setScreen("Model Comparison");
                          }}
                        >
                          <span>
                            {r.experiment}
                            <small>{r.id}</small>
                          </span>
                          <span>n = {r.models.shortcut.test.n}</span>
                          <span>{new Date(r.timestamp).toLocaleString()}</span>
                          <ArrowUpRight size={17} />
                        </button>
                      ))
                    ) : (
                      <Notice>
                        No trained models. Run the training commands above after
                        preparing data.
                      </Notice>
                    )}
                  </section>
                </>
              )}
              {screen === "Model Comparison" &&
                (run ? (
                  <Comparison run={run} openTarget={openTarget} />
                ) : (
                  <Notice>
                    Train this experiment through the CLI to compute its
                    results.
                  </Notice>
                ))}
              {screen === "Signal Explorer" && (
                <Explorer
                  run={run}
                  target={target}
                  setTarget={setTarget}
                  discovery={discovery}
                  setDiscovery={setDiscovery}
                />
              )}
              {screen === "Shortcut Audit" && (
                <Audit manifest={manifest} run={run} openTarget={openTarget} />
              )}
            </>
          )}
          <footer>
            TransitShortcut / locally reproducible evidence
            <span>Dataset auditing · not planet validation</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
function Filters({
  mission,
  setMission,
  truth,
  setTruth,
  classes,
}: {
  mission: string;
  setMission: (v: string) => void;
  truth: string;
  setTruth: (v: string) => void;
  classes: string[];
}) {
  return (
    <>
      <label>
        Mission
        <select value={mission} onChange={(e) => setMission(e.target.value)}>
          <option value="">All missions</option>
          {["kepler", "k2", "tess"].map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      </label>
      <label>
        True disposition
        <select value={truth} onChange={(e) => setTruth(e.target.value)}>
          <option value="">All classes</option>
          {classes.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </label>
    </>
  );
}
function Comparison({
  run,
  openTarget,
}: {
  run: Run;
  openTarget: (id: string) => void;
}) {
  const [mission, setMission] = useState(""),
    [truth, setTruth] = useState(""),
    [pred, setPred] = useState(""),
    [correct, setCorrect] = useState(""),
    [disagree, setDisagree] = useState(false),
    [model, setModel] = useState("shortcut"),
    [offset, setOffset] = useState(0);
  const params = new URLSearchParams({
    mission,
    true_class: truth,
    predicted_class: pred,
    model,
    offset: String(offset),
  });
  if (correct) params.set("correct", correct);
  if (disagree) params.set("disagreement", "true");
  const page = useApi<Page>(`/experiments/${run.id}/predictions?${params}`);
  useEffect(
    () => setOffset(0),
    [mission, truth, pred, correct, disagree, model, run.id],
  );
  const mm = page.value?.metrics;
  return (
    <>
      <Notice>
        {run.experiment === "multiclass"
          ? "Three-class task. Mission filters can remove class support; absent-class recalls are unavailable."
          : "Separate binary task: PLANET versus FALSE_POSITIVE. All missions share the same eligible training cohort; this is not a held-out-mission experiment."}{" "}
        Test results describe this sampled cohort. Validation was evaluated
        separately; no hyperparameter tuning was performed.
      </Notice>
      <div className="toolbar filters">
        <Filters
          {...{ mission, setMission, truth, setTruth }}
          classes={run.classes}
        />
        <label>
          Filter model
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="shortcut">Shortcut Model</option>
            <option value="signal">Signal Model</option>
          </select>
        </label>
        <label>
          Predicted class
          <select value={pred} onChange={(e) => setPred(e.target.value)}>
            <option value="">All predictions</option>
            {run.classes.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <label>
          Outcome
          <select value={correct} onChange={(e) => setCorrect(e.target.value)}>
            <option value="">All outcomes</option>
            <option value="true">Correct</option>
            <option value="false">Incorrect</option>
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={disagree}
            onChange={(e) => setDisagree(e.target.checked)}
          />
          Models disagree
        </label>
      </div>
      {page.loading ? (
        <Notice>Loading filtered evaluation…</Notice>
      ) : page.error ? (
        <Notice>{page.error}</Notice>
      ) : (
        <>
          <div className="grid two">
            {["shortcut", "signal"].map((name) => (
              <section key={name} className={name}>
                <div className="model-title">
                  <span className="model-mark">
                    {name === "shortcut" ? "A" : "B"}
                  </span>
                  <h2>{name === "shortcut" ? "Shortcut" : "Signal"} Model</h2>
                  <span className="badge">LOGISTIC REGRESSION</span>
                </div>
                <p>
                  {name === "shortcut"
                    ? "Mission, cadence, parameter availability, folded zero indicators, and catalog-flag missingness. No flag values or label proxies."
                    : "Features exclusively from raw flux. No mission, catalog parameters, folded availability, or provenance fields."}
                </p>
                <div className="metrics">
                  {[
                    ["Accuracy", "accuracy"],
                    ["Balanced acc.", "balanced_accuracy"],
                    ["Macro F1", "macro_f1"],
                    ["Weighted F1", "weighted_f1"],
                  ].map(([label, key]) => (
                    <div key={key}>
                      <small>{label}</small>
                      <strong>
                        {pct(
                          mm?.[name][key as keyof Metric] as number | undefined,
                        )}
                      </strong>
                    </div>
                  ))}
                </div>
                <p className="mono muted">
                  TRAIN n={run.models[name].train_n} / FILTERED TEST n=
                  {mm?.[name].n ?? 0}
                </p>
                {mm?.[name].warning && <Notice>{mm[name].warning}</Notice>}
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>n</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(mm?.[name].per_class ?? {}).map(
                        ([c, v]) => (
                          <tr key={c}>
                            <td>{nice(c)}</td>
                            <td>{pct(v.precision)}</td>
                            <td>{pct(v.recall)}</td>
                            <td>{pct(v.f1)}</td>
                            <td>{v.support}</td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
                <Chart
                  title={`${name} · confusion matrix (counts)`}
                  xTitle="Predicted class"
                  yTitle="True class"
                  data={[
                    {
                      type: "heatmap",
                      z: mm?.[name].confusion_matrix ?? [],
                      x: run.classes.map(nice),
                      y: run.classes.map(nice),
                      colorscale: [
                        [0, "#111d2b"],
                        [1, name === "shortcut" ? amber : cyan],
                      ],
                      showscale: true,
                      hovertemplate:
                        "True: %{y}<br>Predicted: %{x}<br>n=%{z}<extra></extra>",
                    },
                  ]}
                />
                <details>
                  <summary>Feature allowlist & class support</summary>
                  <p className="mono">{run.models[name].features.join(", ")}</p>
                  <pre>{JSON.stringify(mm?.[name], null, 2)}</pre>
                  <p>
                    Validation macro F1:{" "}
                    {pct(run.models[name].validation.macro_f1)} (n=
                    {run.models[name].validation.n})
                  </p>
                </details>
              </section>
            ))}
          </div>
          <section>
            <h2>Filtered performance comparison</h2>
            <Chart
              title="Both baselines · same filtered evaluation rows"
              data={["shortcut", "signal"].map((name, i) => ({
                type: "bar",
                name: name === "shortcut" ? "A · Shortcut" : "B · Signal",
                x: ["Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1"],
                y: [
                  mm?.[name].accuracy ?? null,
                  mm?.[name].balanced_accuracy ?? null,
                  mm?.[name].macro_f1 ?? null,
                  mm?.[name].weighted_f1 ?? null,
                ],
                marker: { color: i ? cyan : amber },
              }))}
            />
          </section>
          <section>
            <h2>Performance across missions</h2>
            {run.experiment === "cross-mission" ? (
              <Chart
                title="Binary task · macro F1 by mission (unfiltered test cohort)"
                data={["shortcut", "signal"].map((name, i) => ({
                  type: "bar",
                  name: name === "shortcut" ? "A · Shortcut" : "B · Signal",
                  x: ["kepler", "k2", "tess"].map(
                    (m) => `${m} (n=${run.models[name].missions[m].n})`,
                  ),
                  y: ["kepler", "k2", "tess"].map((m) =>
                    run.models[name].missions[m].warning
                      ? null
                      : (run.models[name].missions[m].macro_f1 ?? null),
                  ),
                  marker: { color: i ? cyan : amber },
                }))}
              />
            ) : (
              <p>
                Switch to the binary experiment for a comparable mission chart.
              </p>
            )}
            {run.experiment === "cross-mission" &&
              ["kepler", "k2", "tess"].map(
                (m) =>
                  run.models.shortcut.missions[m].warning && (
                    <Notice key={m}>
                      {m}: {run.models.shortcut.missions[m].warning}; n=
                      {run.models.shortcut.missions[m].n}. Chart value withheld.
                    </Notice>
                  ),
              )}
          </section>
          <section>
            <h2>
              Errors and disagreements <small>n={page.value?.total}</small>
            </h2>
            <div className="errorgrid">
              {Object.entries(page.value?.error_breakdown ?? {}).map(
                ([k, v]) => (
                  <div key={k}>
                    <strong>{v}</strong>
                    <small>{nice(k)}</small>
                  </div>
                ),
              )}
            </div>
            <p className="muted">
              Disagreement overlaps correctness categories. Different errors
              means both wrong with different predicted labels.
            </p>
            <Rows rows={page.value?.items ?? []} openTarget={openTarget} />
            <Pager {...{ offset, setOffset }} total={page.value?.total ?? 0} />
          </section>
        </>
      )}
    </>
  );
}
function Rows({
  rows,
  openTarget,
}: {
  rows: Row[];
  openTarget: (id: string) => void;
}) {
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Target</th>
            <th>Mission / split</th>
            <th>True class</th>
            <th>Shortcut prediction</th>
            <th>Signal prediction</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.target_id}>
              <td>
                <button
                  className="text-button"
                  onClick={() => openTarget(r.target_id)}
                >
                  {r.star_id} ↗
                </button>
              </td>
              <td>
                {r.mission} / {r.split}
              </td>
              <td>{nice(r.disposition)}</td>
              <td className="amber">
                {nice(String(r.shortcut_prediction ?? "—"))}
              </td>
              <td className="cyan">
                {nice(String(r.signal_prediction ?? "—"))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && <Notice>No matching observations.</Notice>}
    </div>
  );
}
function Pager({
  offset,
  setOffset,
  total,
}: {
  offset: number;
  setOffset: (v: number) => void;
  total: number;
}) {
  return (
    <div className="pager">
      <span>
        {total ? offset + 1 : 0}–{Math.min(offset + 50, total)} / {total}
      </span>
      <button
        disabled={!offset}
        onClick={() => setOffset(Math.max(0, offset - 50))}
      >
        Previous
      </button>
      <button
        disabled={offset + 50 >= total}
        onClick={() => setOffset(offset + 50)}
      >
        Next
      </button>
    </div>
  );
}
function Explorer({
  run,
  target,
  setTarget,
  discovery,
  setDiscovery,
}: {
  run?: Run;
  target: string;
  setTarget: (v: string) => void;
  discovery: string;
  setDiscovery: (v: string) => void;
}) {
  const [mission, setMission] = useState(""),
    [truth, setTruth] = useState(""),
    [query, setQuery] = useState(""),
    [pattern, setPattern] = useState("all"),
    [offset, setOffset] = useState(0),
    [view, setView] = useState("flux_raw");
  const params = new URLSearchParams({
    mission,
    true_class: truth,
    query,
    pattern,
    offset: String(offset),
  });
  const listing = useApi<Page>(
    discovery && run
      ? `/experiments/${run.id}/predictions?${new URLSearchParams({ mission, true_class: truth, offset: String(offset), ...(discovery === "disagree" ? { disagreement: "true" } : discovery ? { discovery } : {}) })}`
      : `/targets?${params}`,
  );
  useEffect(() => setOffset(0), [mission, truth, query, pattern, discovery]);
  const detail = useApi<{ data: Target }>(target ? `/targets/${target}` : null),
    flux = useApi<{ data: Flux }>(
      target ? `/targets/${target}/flux?view=${view}` : null,
    ),
    predictions = useApi<{ data: Row }>(
      target && run ? `/targets/${target}/predictions?run_id=${run.id}` : null,
    );
  const rows = listing.value?.items ?? [];
  const d = detail.value?.data,
    f = flux.value?.data,
    p = predictions.value?.data;
  return (
    <>
      <div className="toolbar filters">
        <label>
          Target search
          <input
            value={query}
            placeholder="Star identifier"
            onChange={(e) => {
              setQuery(e.target.value);
              setDiscovery("");
            }}
          />
        </label>
        <Filters
          {...{ mission, setMission, truth, setTruth }}
          classes={["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]}
        />
        <label>
          Artifact pattern
          <select
            value={pattern}
            onChange={(e) => {
              setPattern(e.target.value);
              setDiscovery("");
            }}
          >
            <option value="all">All observations</option>
            <option value="zero">Zero-filled global view</option>
            <option value="incomplete">Incomplete transit parameters</option>
          </select>
        </label>
        <label>
          Evaluated examples
          <select
            value={discovery}
            onChange={(e) => {
              setDiscovery(e.target.value);
              setQuery("");
              setPattern("all");
            }}
          >
            <option value="">All prepared targets</option>
            <option value="disagree">Models disagree</option>
            <option value="shortcut-only">
              Shortcut correct / Signal wrong
            </option>
            <option value="signal-only">Signal correct / Shortcut wrong</option>
            <option value="both-wrong">Both wrong</option>
          </select>
        </label>
      </div>
      <div className="explorer">
        <section className="targetlist">
          <h3>OBSERVATION INDEX</h3>
          {listing.loading ? (
            <p>Loading targets…</p>
          ) : listing.error ? (
            <Notice>{listing.error}</Notice>
          ) : (
            <>
              {rows.map((r) => (
                <button
                  key={r.target_id}
                  className={target === r.target_id ? "selected" : ""}
                  onClick={() => setTarget(r.target_id)}
                >
                  <strong>{r.star_id}</strong>
                  <small>
                    {r.mission} · {r.split}
                  </small>
                  <span>{nice(r.disposition)}</span>
                </button>
              ))}
              {!rows.length && <p>No matching targets.</p>}
              <Pager
                {...{ offset, setOffset }}
                total={listing.value?.total ?? 0}
              />
            </>
          )}
        </section>
        <div>
          {!target ? (
            <section className="empty">
              <ScanLine size={32} />
              <h2>Select an observation</h2>
              <p>
                Light curves load on demand. Raw views preserve local extrema
                during downsampling.
              </p>
            </section>
          ) : detail.loading ? (
            <Notice>Loading selected target…</Notice>
          ) : detail.error ? (
            <Notice>{detail.error}</Notice>
          ) : (
            d && (
              <>
                <section>
                  <div className="heading">
                    <h2 className="mono">{String(d.metadata.star_id)}</h2>
                    <span className="badge">
                      {String(d.metadata.mission)} / {String(d.metadata.split)}
                    </span>
                  </div>
                  <p>
                    Catalog disposition:{" "}
                    <strong>{nice(String(d.metadata.disposition))}</strong>
                  </p>
                  <div className="tabs">
                    {[
                      "flux_raw",
                      "flux_global",
                      "flux_local",
                      "flux_odd",
                      "flux_even",
                    ].map((v) => (
                      <button
                        className={view === v ? "active" : ""}
                        key={v}
                        onClick={() => setView(v)}
                      >
                        {nice(v.replace("flux_", ""))}
                      </button>
                    ))}
                  </div>
                  {flux.loading ? (
                    <Notice>Loading photometry…</Notice>
                  ) : flux.error ? (
                    <Notice>{flux.error}</Notice>
                  ) : f?.unavailable ? (
                    <div className="missing-curve">
                      <ScanLine size={25} />
                      <h3>Folded representation unavailable</h3>
                      <p>{f.annotation}</p>
                      <small>Raw photometry remains available.</small>
                    </div>
                  ) : (
                    f && (
                      <Chart
                        title={`${nice(view)} · ${f.x.length.toLocaleString()} of ${f.original_n.toLocaleString()} samples`}
                        xTitle={f.x_label}
                        yTitle="Relative flux"
                        height={360}
                        data={[
                          {
                            type: "scattergl",
                            mode: "lines",
                            x: f.x,
                            y: f.y,
                            line: { color: cyan, width: 1 },
                            name: "Observed flux",
                            connectgaps: false,
                          },
                        ]}
                      />
                    )
                  )}
                  <p className="muted">
                    Raw index does not imply uniform physical time. Folded bins
                    indicate relative position; no calibrated time axis is
                    inferred.
                  </p>
                </section>
                <section>
                  <h2>Prediction evidence</h2>
                  {p ? (
                    <>
                      <div className="grid two">
                        {["shortcut", "signal"].map((name) => (
                          <div className={name} key={name}>
                            <h3>
                              {name === "shortcut"
                                ? "A · Shortcut Model"
                                : "B · Signal Model"}
                            </h3>
                            <strong>
                              {nice(String(p[`${name}_prediction`]))}
                            </strong>
                            <p>
                              {p[`${name}_prediction`] === p.disposition
                                ? "Correct"
                                : "Incorrect"}{" "}
                              against catalog label
                            </p>
                            {run?.classes.map((c) => (
                              <div className="score" key={c}>
                                <span>{nice(c)}</span>
                                <strong>
                                  {pct(Number(p[`${name}_score_${c}`]))}
                                </strong>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                      <small>Uncalibrated model probability estimates.</small>
                      {run?.ablations.map((a) => (
                        <p key={a.index}>
                          Without {a.group}:{" "}
                          <strong>
                            {nice(String(p[`ablation_${a.index}_prediction`]))}
                          </strong>{" "}
                          ·{" "}
                          {p[`ablation_${a.index}_prediction`] !==
                          p.shortcut_prediction
                            ? "prediction changed"
                            : "prediction unchanged"}{" "}
                          · max class-score change{" "}
                          {pct(Number(p[`ablation_${a.index}_score_delta`]))}
                        </p>
                      ))}
                    </>
                  ) : (
                    <Notice>
                      {predictions.loading
                        ? "Loading predictions…"
                        : (predictions.error ??
                          "Train an experiment to inspect evaluated test predictions.")}
                    </Notice>
                  )}
                </section>
                <section>
                  <h2>Feature values & provenance</h2>
                  <details open>
                    <summary>Shortcut inputs actually used</summary>
                    <div className="featuregrid">
                      {Object.entries(d.shortcut_features).map(([k, v]) => (
                        <div key={k}>
                          <span>{nice(k)}</span>
                          <code>{v}</code>
                        </div>
                      ))}
                    </div>
                  </details>
                  <details>
                    <summary>Raw-photometry features</summary>
                    <pre>{JSON.stringify(d.signal_features, null, 2)}</pre>
                  </details>
                  <details>
                    <summary>Catalog metadata / source provenance</summary>
                    <pre>{JSON.stringify(d.metadata, null, 2)}</pre>
                  </details>
                </section>
              </>
            )
          )}
        </div>
      </div>
    </>
  );
}
function Audit({
  manifest,
  run,
  openTarget,
}: {
  manifest: Manifest;
  run?: Run;
  openTarget: (id: string) => void;
}) {
  const [ablation, setAblation] = useState(0),
    [mission, setMission] = useState("");
  const rows = useApi<Page>(
    run
      ? `/experiments/${run.id}/predictions?limit=10&sort_ablation=${ablation}`
      : null,
  );
  const groups = manifest.summary.groups.filter(
    (g) => !mission || g.mission === mission,
  );
  const a = run?.ablations.find((a) => a.index === ablation);
  const ranked = [...(rows.value?.items ?? [])].sort(
    (a, b) =>
      Number(b[`ablation_${ablation}_score_delta`]) -
      Number(a[`ablation_${ablation}_score_delta`]),
  );
  return (
    <>
      <Notice>
        Observed association is a property of this dataset. Dropping a feature
        group and refitting measures this pipeline's sensitivity; correlated
        remaining features can preserve shortcuts. Neither result establishes
        causal or physical understanding.
      </Notice>
      <section>
        <h2>Artifact inventory</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Feature group</th>
                <th>Why suspicious</th>
                <th>Measured prevalence / association</th>
                <th>Model evidence</th>
              </tr>
            </thead>
            <tbody>
              {[
                [
                  "Mission/cadence",
                  "Mission coverage differs by class",
                  JSON.stringify(manifest.summary.missions),
                ],
                [
                  "Transit availability",
                  "Catalog availability follows classification",
                  `${manifest.summary.artifact_counts.missing_transit} missing / ${Object.values(manifest.actual_sizes).reduce((a, b) => a + b, 0)}`,
                ],
                [
                  "Folded zeros",
                  "Missing representations are encoded as zeros",
                  `${manifest.summary.artifact_counts.zero_folded} all-zero folded rows`,
                ],
                [
                  "Flag missingness",
                  "Catalog vetting coverage varies by mission and label",
                  `${manifest.summary.artifact_counts.missing_flags} rows with all flags absent`,
                ],
              ].map(([g, why, prev]) => (
                <tr key={g}>
                  <td>{g}</td>
                  <td>{why}</td>
                  <td>{prev}</td>
                  <td>
                    {run
                      ? `Sensitivity measured · ΔF1 ${pct(run.ablations.find((a) => a.group === g)?.macro_f1_drop)}`
                      : "Not evaluated"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="grid two">
        <section>
          <h2>Refit feature-group ablations</h2>
          {run ? (
            <Chart
              title="Test macro F1 · original versus refitted model"
              data={[
                {
                  type: "bar",
                  name: "Original Shortcut Model",
                  x: run.ablations.map((a) => a.group),
                  y: run.ablations.map(
                    () => run.models.shortcut.test.macro_f1 ?? null,
                  ),
                  marker: { color: amber },
                },
                {
                  type: "bar",
                  name: "Group removed + refit",
                  x: run.ablations.map((a) => a.group),
                  y: run.ablations.map((a) => a.metrics.macro_f1 ?? null),
                  marker: { color: "#72869a" },
                },
              ]}
            />
          ) : (
            <Notice>
              No completed ablation. Train a model to run the audit.
            </Notice>
          )}
        </section>
        <section>
          <div className="heading">
            <h2>Artifact prevalence</h2>
            <select
              aria-label="Audit mission"
              value={mission}
              onChange={(e) => setMission(e.target.value)}
            >
              <option value="">All missions</option>
              {["kepler", "k2", "tess"].map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>
          <Chart
            title="Within-group fraction · absent mission/classes omitted"
            height={340}
            data={[
              {
                type: "heatmap",
                z: groups.map((g) => [
                  g.missing_transit,
                  g.zero_folded,
                  g.missing_flags,
                ]),
                x: ["Missing params", "Zero folded", "Missing flags"],
                y: groups.map(
                  (g) => `${g.mission} / ${nice(g.class)} (n=${g.n})`,
                ),
                zmin: 0,
                zmax: 1,
                colorscale: [
                  [0, "#14202c"],
                  [1, amber],
                ],
                hovertemplate: "%{y}<br>%{x}: %{z:.1%}<extra></extra>",
              },
            ]}
          />
        </section>
      </div>
      {run && (
        <section>
          <div className="heading">
            <h2>Individual sensitivity</h2>
            <select
              aria-label="Ablation"
              value={ablation}
              onChange={(e) => setAblation(Number(e.target.value))}
            >
              {run.ablations.map((a) => (
                <option key={a.index} value={a.index}>
                  {a.group}
                </option>
              ))}
            </select>
          </div>
          <p>
            {a?.changed_n} / {a?.metrics.n} changed predictions (
            {pct(a?.changed_fraction)}). Mean maximum class-score change:{" "}
            {pct(a?.mean_max_score_delta)}.
          </p>
          <div className="grid two">
            <pre>
              BY MISSION{"\n"}
              {JSON.stringify(a?.by_mission, null, 2)}
            </pre>
            <pre>
              BY TRUE CLASS{"\n"}
              {JSON.stringify(a?.by_class, null, 2)}
            </pre>
          </div>
          <p className="muted">
            Largest score changes across the complete evaluated cohort. Export
            CSV for all rows.
          </p>
          <Rows rows={ranked.slice(0, 10)} openTarget={openTarget} />
          <div className="toolbar">
            {["json", "markdown", "csv"].map((f) => (
              <a
                key={f}
                className="button"
                href={`/api/reports/${run.id}?format=${f}`}
              >
                <Download size={15} />
                {f.toUpperCase()}
              </a>
            ))}
          </div>
        </section>
      )}
      <section>
        <h2>Interpretation boundaries</h2>
        {(
          run?.limitations ?? ["Prepare and train models to measure reliance."]
        ).map((l) => (
          <p key={l}>{l}</p>
        ))}
      </section>
    </>
  );
}
const mount = document.getElementById("root");
if (mount) createRoot(mount).render(<App />);
export { App };
