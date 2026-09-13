"use client";

import { useState } from "react";
import { Case } from "@/lib/types";
import { decideCase, imageUrl } from "@/lib/api";
import StatusPill from "./StatusPill";
import {
  MousePointerClick,
  Tag,
  FileText,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  Loader2,
  Copy,
  Check,
} from "lucide-react";

type Props = {
  activeCase: Case | null;
  onResolved: (threadId: string) => void;
};

function verdictTone(verdict: Case["consistency_verdict"]) {
  if (verdict === "INCONSISTENT") return "brick" as const;
  if (verdict === "CONSISTENT") return "sage" as const;
  return "amber" as const;
}

function verdictGlow(verdict: Case["consistency_verdict"]) {
  if (verdict === "INCONSISTENT") return "shadow-[0_0_0_1px_rgba(208,96,90,0.25),0_8px_30px_-8px_rgba(208,96,90,0.35)]";
  if (verdict === "CONSISTENT") return "shadow-[0_0_0_1px_rgba(99,169,134,0.25),0_8px_30px_-8px_rgba(99,169,134,0.35)]";
  return "shadow-[0_0_0_1px_rgba(224,169,64,0.25),0_8px_30px_-8px_rgba(224,169,64,0.35)]";
}

function similarityPercent(distance: number): number {
  const clamped = Math.max(0, Math.min(distance, 2));
  return Math.round((1 - clamped / 2) * 100);
}

export default function CaseDetail({ activeCase, onResolved }: Props) {
  const [submitting, setSubmitting] = useState<"confirmed_fraud" | "false_positive" | null>(null);
  const [copied, setCopied] = useState(false);

  if (!activeCase) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[#1a1e27] ring-1 ring-[#2a2f3c]">
          <MousePointerClick size={20} className="text-[#8b92a3]" strokeWidth={1.75} />
        </div>
        <p className="text-sm text-[#8b92a3]">Select a case from the queue to review it.</p>
      </div>
    );
  }

  const handleDecision = async (decision: "confirmed_fraud" | "false_positive") => {
    setSubmitting(decision);
    try {
      await decideCase(activeCase.thread_id, decision);
      onResolved(activeCase.thread_id);
    } finally {
      setSubmitting(null);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCase.thread_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div key={activeCase.thread_id} className="flex h-full flex-col overflow-y-auto p-6 animate-fade-in">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold text-[#eceef2]">{activeCase.listing_title}</h2>
          <button
            onClick={handleCopy}
            className="mt-1 flex items-center gap-1.5 font-mono text-xs text-[#6b7385] transition-colors hover:text-[#8b92a3]"
          >
            {copied ? <Check size={12} className="text-[#63a986]" /> : <Copy size={12} />}
            {activeCase.thread_id} - seller: {activeCase.seller_id}
          </button>
        </div>
        <StatusPill
          label={activeCase.consistency_verdict ?? "UNCERTAIN"}
          tone={verdictTone(activeCase.consistency_verdict)}
        />
      </div>

      <a
        href={imageUrl(activeCase.thread_id)}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          height: "320px",
          resize: "vertical" as const,
          background: "radial-gradient(circle at center, #14171f 0%, #0b0d11 75%)",
        }}
        className={`group relative mt-5 flex min-h-[160px] max-h-[800px] w-full cursor-zoom-in items-center justify-center overflow-auto rounded-xl ring-1 ring-[#2a2f3c] transition-shadow hover:ring-[#3a4152] ${verdictGlow(
          activeCase.consistency_verdict
        )}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl(activeCase.thread_id)}
          alt={activeCase.listing_title}
          className="h-full w-full object-contain"
        />
        <span className="pointer-events-none absolute bottom-2 right-2 rounded-md bg-[#101319]/80 px-2 py-1 text-[10px] text-[#8b92a3] opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
          drag corner to resize · click to expand
        </span>
      </a>

      <div className="mt-5 grid grid-cols-3 gap-3">
        <div className="rounded-lg bg-[#1a1e27] p-3.5 ring-1 ring-[#21252f]">
          <div className="flex items-center gap-1.5 text-[#6b7385]">
            <Tag size={12} strokeWidth={2} />
            <p className="text-xs">Category</p>
          </div>
          <p className="mt-1 text-sm text-[#eceef2]">{activeCase.listing_category || "-"}</p>
        </div>
        <div className="col-span-2 rounded-lg bg-[#1a1e27] p-3.5 ring-1 ring-[#21252f]">
          <div className="flex items-center gap-1.5 text-[#6b7385]">
            <FileText size={12} strokeWidth={2} />
            <p className="text-xs">Listing description</p>
          </div>
          <p className="mt-1 text-sm text-[#eceef2]">{activeCase.listing_description || "-"}</p>
        </div>
      </div>

      <div className="mt-5 rounded-lg bg-[#1a1e27] p-3.5 ring-1 ring-[#21252f]">
        <div className="flex items-center gap-1.5 text-[#6b7385]">
          <ShieldAlert size={12} strokeWidth={2} />
          <p className="text-xs">Consistency reasoning</p>
        </div>
        <p className="mt-1.5 text-sm leading-relaxed text-[#eceef2]">
          {activeCase.consistency_reason ?? "No reasoning available."}
        </p>
      </div>

      <div className="mt-5">
        <div className="flex items-center gap-1.5 text-[#6b7385]">
          <ScanSearch size={12} strokeWidth={2} />
          <p className="text-xs">Matched listings ({activeCase.matches.length})</p>
        </div>
        <ul className="mt-2 space-y-1.5">
          {activeCase.matches.map((m) => {
            const pct = similarityPercent(m.l2_distance);
            return (
              <li
                key={m.listing_id}
                className="rounded-lg bg-[#1a1e27] px-3.5 py-2.5 ring-1 ring-[#21252f]"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-[#eceef2]">{m.listing_id}</span>
                  <span className="font-mono text-[#6b7385]">seller: {m.seller_id}</span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#0b0d11]">
                    <div
                      className={`h-full rounded-full ${pct >= 90 ? "bg-[#d0605a]" : pct >= 60 ? "bg-[#e0a940]" : "bg-[#63a986]"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-[#8b92a3] w-10 text-right">{pct}%</span>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="mt-6 flex gap-3 border-t border-[#21252f] pt-5">
        <button
          onClick={() => handleDecision("confirmed_fraud")}
          disabled={submitting !== null}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-[#d0605a] px-4 py-2.5 text-sm font-medium text-[#101319] transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-50"
        >
          {submitting === "confirmed_fraud" ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <ShieldAlert size={15} />
          )}
          Confirm fraud
        </button>
        <button
          onClick={() => handleDecision("false_positive")}
          disabled={submitting !== null}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-transparent px-4 py-2.5 text-sm font-medium text-[#eceef2] ring-1 ring-[#2a2f3c] transition-all hover:bg-[#1a1e27] active:scale-[0.98] disabled:opacity-50"
        >
          {submitting === "false_positive" ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <ShieldCheck size={15} />
          )}
          False positive
        </button>
      </div>
    </div>
  );
}
