export type Match = {
  listing_id: string;
  seller_id: string;
  l2_distance: number;
};

export type Case = {
  thread_id: string;
  listing_id: string;
  seller_id: string;
  listing_title: string;
  listing_description: string;
  listing_category: string;
  image_filename: string;
  flagged: number;
  matches: Match[];
  consistency_verdict: "CONSISTENT" | "INCONSISTENT" | "UNCERTAIN" | null;
  consistency_reason: string | null;
  human_decision: "confirmed_fraud" | "false_positive" | null;
  created_at: string;
};
