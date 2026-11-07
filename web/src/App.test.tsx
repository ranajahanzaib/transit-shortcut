// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "./main";
vi.mock("plotly.js-dist-min", () => ({
  default: { react: vi.fn(), purge: vi.fn(), Plots: { resize: vi.fn() } },
}));
const manifest = {
  dataset_id: "fixture",
  revision: "fixture",
  actual_sizes: { train: 3, test: 3, validation: 3 },
  config: { mode: "sample", seed: 42 },
  summary: {
    classes: { PLANET: 3 },
    missions: { kepler: 9 },
    groups: [
      {
        mission: "kepler",
        class: "PLANET",
        n: 9,
        missing_transit: 0,
        zero_folded: 0,
        missing_flags: 0,
      },
    ],
    artifact_counts: { zero_folded: 0, missing_transit: 0, missing_flags: 0 },
  },
};
function reply(value: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(value) });
}
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
it("renders setup without fabricating metrics", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn((path: string) =>
      path.includes("summary")
        ? Promise.resolve({
            ok: false,
            json: () => Promise.resolve({ detail: "Dataset not prepared" }),
          })
        : reply({ data: [] }),
    ),
  );
  render(<App />);
  expect(
    await screen.findByText("Begin with an auditable sample."),
  ).toBeInTheDocument();
  expect(screen.queryByText("99.0%")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Model Comparison" }));
  expect(
    screen.getByRole("heading", { name: "Model Comparison", level: 1 }),
  ).toBeInTheDocument();
});
it("renders pages, missing models, selects an actual fixture target and annotates missing folded data", async () => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "fetch",
    vi.fn((path: string) => {
      if (path.includes("summary")) return reply({ data: manifest });
      if (path.includes("/experiments")) return reply({ data: [] });
      if (path.includes("/flux"))
        return reply({
          data: {
            x: [0, 1],
            y: [1, 1],
            original_n: 2,
            x_label: "Sample index",
            unavailable: path.includes("global"),
            annotation: "Stored zeros are not observed flux",
          },
        });
      if (path.includes("/targets/test-0"))
        return reply({
          data: {
            metadata: {
              star_id: "SYNTHETIC_TEST",
              mission: "kepler",
              split: "test",
              disposition: "PLANET",
            },
            shortcut_features: { flux_global_zero: 1 },
            signal_features: { std: 0.1 },
          },
        });
      return reply({
        total: 1,
        items: [
          {
            target_id: "test-0",
            star_id: "SYNTHETIC_TEST",
            mission: "kepler",
            split: "test",
            disposition: "PLANET",
          },
        ],
      });
    }),
  );
  render(<App />);
  await screen.findByText("Construction traces");
  fireEvent.click(screen.getByRole("button", { name: "Model Comparison" }));
  expect(
    await screen.findByText(
      "Train this experiment through the CLI to compute its results.",
    ),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Signal Explorer" }));
  fireEvent.click(
    await screen.findByRole("button", { name: /SYNTHETIC_TEST/ }),
  );
  await screen.findByRole("heading", { name: "SYNTHETIC_TEST" });
  fireEvent.click(screen.getByRole("button", { name: "global" }));
  expect(
    await screen.findByText("Folded representation unavailable"),
  ).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Shortcut Audit" }));
  expect(await screen.findByText("Artifact inventory")).toBeInTheDocument();
});
it("shows loading state while data are pending", () => {
  vi.stubGlobal("fetch", () => new Promise(() => {}));
  render(<App />);
  expect(
    screen.getByText("Loading local dataset artifacts…"),
  ).toBeInTheDocument();
});
it("switches completed experiments and model filters", async () => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  const metric = {
    n: 6,
    available: true,
    accuracy: 0.5,
    balanced_accuracy: 0.5,
    macro_f1: 0.5,
    weighted_f1: 0.5,
    support: { PLANET: 3, FALSE_POSITIVE: 3 },
    confusion_matrix: [
      [2, 1],
      [2, 1],
    ],
    class_order: ["PLANET", "FALSE_POSITIVE"],
  };
  const model = {
    train_n: 12,
    test: metric,
    validation: metric,
    features: ["fixture"],
    missions: { kepler: metric, k2: metric, tess: metric },
  };
  const runs = ["multiclass", "cross-mission"].map((experiment) => ({
    id: `${experiment}-fixture`,
    experiment,
    dataset_id: "fixture",
    timestamp: "2026-01-01T00:00:00Z",
    audit_completed_at: "2026-01-01T00:00:00Z",
    classes: ["PLANET", "FALSE_POSITIVE"],
    models: { shortcut: model, signal: model },
    ablations: [],
    limitations: [],
  }));
  const fetcher = vi.fn((path: string) =>
    path.includes("summary")
      ? reply({ data: manifest })
      : path.includes("predictions")
        ? reply({
            total: 6,
            items: [],
            metrics: { shortcut: metric, signal: metric },
            error_breakdown: {
              both_correct: 3,
              both_wrong: 3,
              disagree: 0,
              different_errors: 0,
            },
          })
        : reply({ data: runs }),
  );
  vi.stubGlobal("fetch", fetcher);
  render(<App />);
  await screen.findByText("Construction traces");
  fireEvent.click(screen.getByRole("button", { name: "Model Comparison" }));
  await screen.findByText("Errors and disagreements", { exact: false });
  fireEvent.change(screen.getByLabelText("Experiment"), {
    target: { value: "cross-mission" },
  });
  expect(screen.getByLabelText("Completed run")).toHaveValue(
    "cross-mission-fixture",
  );
  fireEvent.change(screen.getByLabelText("Filter model"), {
    target: { value: "signal" },
  });
  await waitFor(() =>
    expect(
      fetcher.mock.calls.some(
        ([path]) =>
          path.includes("cross-mission-fixture/predictions") &&
          path.includes("model=signal"),
      ),
    ).toBe(true),
  );
});
