import { basename } from "path";

/** Durable document metadata used to reproduce the splitter input at embed time. */
export function buildEmbedFrontmatter(
  docTitle: string | null | undefined,
  path: string,
  description: string | null | undefined,
): { title: string; description?: string } {
  const frontmatter: { title: string; description?: string } = {
    title: docTitle || basename(path).replace(/\.(md|txt)$/i, ""),
  };
  if (description) frontmatter.description = description;
  return frontmatter;
}
