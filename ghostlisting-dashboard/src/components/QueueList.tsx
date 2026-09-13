"use client";

import { Case } from "@/lib/types";
import StatusPill from "./StatusPill";
import { Inbox, Image as ImageIcon, ChevronRight } from "lucide-react";
import { imageUrl } from "@/lib/api";

type Props = {
  cases: Case[];
  selectedId: string | null;
  onSelect: (threadId: string) => void;
};

function verdictTone(verdict: Case["consistency_verdict"]) {
  if (verdict === "INCONSISTENT") return "brick" as const;
  if (verdict === "CONSISTENT") return "sage" as const;
  return "amber" as const;
}

export default function QueueList({ cases, selectedId, onSelect }: Props) {
  if (cases.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[#1a1e27] ring-1 ring-[#2a2f3c]">
          <Inbox size={18} className="text-[#8b92a3]" strokeWidth={1.75} />
        </div>
        <p className="text-sm text-[#8b92a3]">
          Queue is empty.<br />New flags will appear here automatically.
        </p>
      </div>
    );
  }

  return (
    <ul className="divide-y divide-[#21252f] p-1.5">
      {cases.map((c) => {
        const isSelected = c.thread_id === selectedId;
        return (
          <li key={c.thread_id} className="my-0.5">
            <button
              onClick={() => onSelect(c.thread_id)}
              className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all duration-150 ${
                isSelected
                  ? "bg-[#1f2430] ring-1 ring-[#2f3646]"
                  : "hover:bg-[#1a1e27]"
              }`}
            >
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-md ring-1 transition-colors ${
                  isSelected ? "ring-[#e0a940]/40" : "ring-[#2a2f3c]"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageUrl(c.thread_id)}
                  alt=""
                  className="h-full w-full object-cover"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              </div>

              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium text-[#eceef2]">
                    {c.listing_title}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <StatusPill label={c.consistency_verdict ?? "UNCERTAIN"} tone={verdictTone(c.consistency_verdict)} size="sm" />
                  <span className="font-mono text-[11px] text-[#6b7385]">{c.seller_id}</span>
                </div>
              </div>

              <ChevronRight
                size={15}
                strokeWidth={2}
                className={`shrink-0 transition-all ${
                  isSelected ? "text-[#e0a940] translate-x-0" : "text-transparent -translate-x-1 group-hover:text-[#4a5164] group-hover:translate-x-0"
                }`}
              />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
