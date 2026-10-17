import { basename } from "path";

/**
 * Durable document metadata used to reproduce the splitter input at embed time.
 *
 * Key order is load-bearing: `splitDocument` emits one `frontmatter` fragment per
 * key in insertion order, so title → description → keywords fixes the seq layout.
 * A key is added only when its value is non-empty, so a document without
 * description/keywords produces exactly the fragment set it did before those
 * columns existed (no re-embed churn). `keywords` is the normalized
 * comma-joined string persisted in `documents.keywords` (master-harness-65amk).
 */
export function buildEmbedFrontmatter(
  docTitle: string | null | undefined,
  path: string,
  description: string | null | undefined,
  keywords?: string | null,
): { title: string; description?: string; keywords?: string } {
  const frontmatter: { title: string; description?: string; keywords?: string } = {
    title: docTitle || basename(path).replace(/\.(md|txt)$/i, ""),
  };
  if (description) frontmatter.description = description;
  if (keywords) frontmatter.keywords = keywords;
  return frontmatter;
}
