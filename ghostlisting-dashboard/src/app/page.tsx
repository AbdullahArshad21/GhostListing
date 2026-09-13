"use client";

import { useEffect, useState, useCallback } from "react";
import { Case } from "@/lib/types";
import { fetchPending, fetchResolved } from "@/lib/api";
import QueueList from "@/components/QueueList";
import CaseDetail from "@/components/CaseDetail";
import { Ghost, Inbox, History, WifiOff } from "lucide-react";

const POLL_INTERVAL_MS = 4000;

export default function Home() {
  const [view, setView] = useState<"pending" | "resolved">("pending");
  const [pending, setPending] = useState<Case[]>([]);
  const [resolved, setResolved] = useState<Case[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [p, r] = await Promise.all([fetchPending(), fetchResolved()]);
      setPending(p);
      setResolved(r);
      setError(null);
    } catch {
      setError("Can't reach the API — is uvicorn running on port 8000?");
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [load]);

  const activeList = view === "pending" ? pending : resolved;
  const activeCase = activeList.find((c) => c.thread_id === selectedId) ?? null;

  const handleResolved = (threadId: string) => {
    setPending((prev) => prev.filter((c) => c.thread_id !== threadId));
    setSelectedId(null);
    load();
  };

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-[#21252f] bg-[#101319]/80 px-6 py-3.5 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-[#e0a940]/20 to-[#63a986]/20 ring-1 ring-[#2a2f3c]">
            <Ghost size={16} className="text-[#e0a940]" strokeWidth={2} />
          </div>
          <div className="flex items-baseline gap-2.5">
            <h1 className="text-sm font-semibold text-[#eceef2]">GhostListing</h1>
            <span className="text-xs text-[#6b7385]">Review queue</span>
          </div>
        </div>

        <nav className="flex gap-1 rounded-lg bg-[#1a1e27] p-1 ring-1 ring-[#21252f]">
          <button
            onClick={() => { setView("pending"); setSelectedId(null); }}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
              view === "pending" ? "bg-[#242a37] text-[#eceef2]" : "text-[#6b7385] hover:text-[#8b92a3]"
            }`}
          >
            <Inbox size={13} strokeWidth={2} />
            Pending
            {pending.length > 0 && (
              <span className="ml-0.5 rounded-full bg-[#e0a940]/20 px-1.5 text-[11px] font-medium text-[#e0a940]">
                {pending.length}
              </span>
            )}
          </button>
          <button
            onClick={() => { setView("resolved"); setSelectedId(null); }}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
              view === "resolved" ? "bg-[#242a37] text-[#eceef2]" : "text-[#6b7385] hover:text-[#8b92a3]"
            }`}
          >
            <History size={13} strokeWidth={2} />
            Resolved
          </button>
        </nav>
      </header>

      {error && (
        <div className="flex items-center gap-2 border-b border-[#21252f] bg-[#d0605a]/8 px-6 py-2 text-sm text-[#d0605a]">
          <WifiOff size={14} />
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 shrink-0 overflow-y-auto border-r border-[#21252f]">
          <QueueList cases={activeList} selectedId={selectedId} onSelect={setSelectedId} />
        </aside>
        <main className="flex-1 overflow-hidden">
          <CaseDetail activeCase={activeCase} onResolved={handleResolved} />
        </main>
      </div>
    </div>
  );
}
