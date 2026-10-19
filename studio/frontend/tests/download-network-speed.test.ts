// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { DownloadTransferProgress } from "../src/features/hub/download-manager/api.ts";
import type { JobRuntime } from "../src/features/hub/download-manager/download-manager-types.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { applySpeedSample } = await import(
  "../src/features/hub/download-manager/download-speed.ts"
);
const MB = 1e6;
const TOTAL = 194_000 * MB;
const runtime = (): Pick<
  JobRuntime,
  "speedSamples" | "networkSamples" | "speedSource"
> => ({
  speedSamples: [],
  networkSamples: [],
});
const transfer = (
  t: number,
  overrides: Partial<DownloadTransferProgress> = {},
): DownloadTransferProgress => ({
  attempt_id: "worker-a",
  network_bytes: t * 70 * MB,
  elapsed_seconds: t,
  age_seconds: 0,
  ...overrides,
});

test("70 MB/s remains visible while disk progress advances only every 90 seconds", () => {
  const rt = runtime();
  for (let t = 0; t <= 180; t += 2) {
    const disk = 24_000 * MB + Math.floor(t / 90) * 6_300 * MB;
    const stats = applySpeedSample(
      rt,
      disk,
      TOTAL,
      t * 1_000 + (t % 4) * 100,
      transfer(t),
    );
    if (t >= 6) assert.ok(Math.abs(stats.bytesPerSec - 70 * MB) < 1);
    assert.equal(stats.etaSeconds, 0);
  }
});

test("repeated polls of one heartbeat do not create a rate", () => {
  const rt = runtime();
  for (let t = 0; t < 10; t++) {
    assert.equal(
      applySpeedSample(rt, 24_000 * MB, TOTAL, t * 1_000, transfer(1))
        .bytesPerSec,
      0,
    );
  }
});

test("stale network statistics hide the rate even if reconstruction continues", () => {
  const rt = runtime();
  for (let t = 0; t <= 10; t++)
    applySpeedSample(rt, 24_000 * MB, TOTAL, t * 1_000, transfer(t));
  const stale = applySpeedSample(
    rt,
    30_000 * MB,
    TOTAL,
    26_000,
    transfer(10, { age_seconds: 16 }),
  );
  assert.deepEqual(stale, { bytesPerSec: 0, etaSeconds: 0 });
  assert.equal(
    applySpeedSample(rt, 30_000 * MB, TOTAL, 27_000, transfer(27)).bytesPerSec,
    0,
  );
});

test("retry and HTTP fallback each start a fresh rate window", () => {
  const rt = runtime();
  for (let t = 0; t <= 10; t++)
    applySpeedSample(rt, 24_000 * MB, TOTAL, t * 1_000, transfer(t));
  // A new worker may resume the same cache without a counter decrease.
  assert.equal(
    applySpeedSample(
      rt,
      24_000 * MB,
      TOTAL,
      11_000,
      transfer(11, { attempt_id: "worker-b" }),
    ).bytesPerSec,
    0,
  );
  assert.equal(
    applySpeedSample(rt, 24_000 * MB, TOTAL, 12_000, null).bytesPerSec,
    0,
  );
  let stats = { bytesPerSec: 0, etaSeconds: 0 };
  for (let t = 13; t <= 20; t++) {
    stats = applySpeedSample(
      rt,
      (24_000 + (t - 12) * 10) * MB,
      TOTAL,
      t * 1_000,
    );
  }
  assert.equal(stats.bytesPerSec, 10 * MB);
  assert.ok(stats.etaSeconds > 0);
});

test("ETA uses file throughput even when network throughput is seven times higher", () => {
  const rt = runtime();
  for (let t = 0; t <= 600; t += 2) {
    const disk = 24_000 * MB + Math.floor(t / 90) * 900 * MB;
    const stats = applySpeedSample(rt, disk, TOTAL, t * 1_000, transfer(t));
    if (t >= 360) {
      assert.equal(stats.bytesPerSec, 70 * MB);
      assert.ok(Math.abs(stats.etaSeconds - (TOTAL - disk) / (10 * MB)) < 1);
    }
  }
});

test("file reconstruction can retain its ETA after network transfer stops", () => {
  const rt = runtime();
  let stats = { bytesPerSec: 0, etaSeconds: 0 };
  for (let t = 0; t <= 30; t++) {
    stats = applySpeedSample(
      rt,
      (24_000 + t * 10) * MB,
      TOTAL,
      t * 1_000,
      transfer(Math.min(t, 10), { age_seconds: Math.max(0, t - 10) }),
    );
  }
  assert.equal(stats.bytesPerSec, 0);
  assert.equal(stats.etaSeconds, (TOTAL - 24_300 * MB) / (10 * MB));
});
