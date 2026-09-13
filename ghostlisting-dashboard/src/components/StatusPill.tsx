import { AlertTriangle, CheckCircle2, HelpCircle, LucideIcon } from "lucide-react";

type Props = {
  label: string;
  tone: "amber" | "brick" | "sage" | "muted";
  size?: "sm" | "md";
};

const toneStyles: Record<Props["tone"], string> = {
  amber: "bg-[#e0a940]/12 text-[#e0a940] ring-1 ring-inset ring-[#e0a940]/25",
  brick: "bg-[#d0605a]/12 text-[#d0605a] ring-1 ring-inset ring-[#d0605a]/25",
  sage: "bg-[#63a986]/12 text-[#63a986] ring-1 ring-inset ring-[#63a986]/25",
  muted: "bg-[#8b92a3]/10 text-[#8b92a3] ring-1 ring-inset ring-[#8b92a3]/20",
};

const toneIcons: Record<Props["tone"], LucideIcon> = {
  amber: HelpCircle,
  brick: AlertTriangle,
  sage: CheckCircle2,
  muted: HelpCircle,
};

export default function StatusPill({ label, tone, size = "md" }: Props) {
  const Icon = toneIcons[tone];
  const padding = size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs";
  const iconSize = size === "sm" ? 11 : 13;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full font-medium tracking-wide ${padding} ${toneStyles[tone]}`}
    >
      <Icon size={iconSize} strokeWidth={2.25} />
      {label}
    </span>
  );
}
