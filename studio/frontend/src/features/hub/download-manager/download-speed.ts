// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  appendSample,
  computeTransferStats,
  STALL_WINDOW_SECONDS,
} from "@/lib/transfer-stats";
import type { JobRuntime, ProgressLike } from "./download-manager-types";

export function applySpeedSample(
  rt: Pick<JobRuntime, "speedSamples" | "networkSamples" | "speedSource">,
  downloadedBytes: number,
  expectedBytes: number,
  nowMs: number,
  transfer?: ProgressLike["transfer_progress"],
): { bytesPerSec: number; etaSeconds: number } {
  const source = transfer ? `network:${transfer.attempt_id}` : "disk";
  if (rt.speedSource !== source) {
    rt.speedSamples.length = 0;
    rt.networkSamples.length = 0;
    rt.speedSource = source;
  }
  // File progress determines ETA independently of compressed or deduplicated network bytes.
  appendSample(rt.speedSamples, nowMs / 1000, downloadedBytes);
  const stats = computeTransferStats(rt.speedSamples, expectedBytes);
  const etaSeconds = stats.stable ? stats.etaSeconds : 0;
  if (transfer) {
    // Use the worker's sampling clock rather than browser polling intervals.
    const {
      network_bytes: bytes,
      elapsed_seconds: elapsed,
      age_seconds: age,
    } = transfer;
    if (age > STALL_WINDOW_SECONDS) {
      rt.networkSamples.length = 0;
      return { bytesPerSec: 0, etaSeconds };
    }
    const last = rt.networkSamples.at(-1);
    if (!last || elapsed > last.t || bytes < last.b) {
      appendSample(rt.networkSamples, elapsed, bytes);
    }
    const network = computeTransferStats(rt.networkSamples, 0);
    return {
      bytesPerSec: network.stable ? network.rateBytesPerSecond : 0,
      etaSeconds,
    };
  }
  return {
    bytesPerSec: stats.stable ? stats.rateBytesPerSecond : 0,
    etaSeconds,
  };
}
