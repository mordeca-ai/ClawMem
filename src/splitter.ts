/**
 * ClawMem Document Splitter — Granular Fragment Indexing
 *
 * Splits markdown documents into semantic fragments (sections, bullet lists,
 * code blocks, frontmatter facts) for per-fragment embedding. Each fragment
 * gets its own vector, dramatically improving recall for specific facts
 * buried in larger documents.
 */

// =============================================================================
// Types
// =============================================================================

export interface Fragment {
  type: 'full' | 'section' | 'list' | 'code' | 'frontmatter' | 'fact' | 'narrative';
  label: string | null;
  content: string;
  startLine: number;
}

// =============================================================================
// Config
// =============================================================================

import { MAX_FRAGMENTS_PER_DOC, MAX_SPLITTER_INPUT_CHARS } from "./limits.ts";

const MIN_FRAGMENT_CHARS = 50;
export const MAX_FRAGMENT_CHARS = 2000;
const FRAGMENT_OVERLAP_CHARS = 200;
const MIN_DOC_CHARS_FOR_SPLIT = 200;

// =============================================================================
// Main Splitter
// =============================================================================

/**
 * Split a markdown document into semantic fragments for embedding.
 * Always includes a 'full' fragment (entire body). Additional fragments
 * are only generated if the document is large enough to benefit from splitting.
 */
export function splitDocument(
  body: string,
  frontmatter?: Record<string, any>
): Fragment[] {
  // Bound input size to prevent memory blowup
  const boundedBody = body.length > MAX_SPLITTER_INPUT_CHARS
    ? body.slice(0, MAX_SPLITTER_INPUT_CHARS)
    : body;

  const fragments: Fragment[] = [];

  // Always include full document as first fragment
  fragments.push({ type: 'full', label: null, content: boundedBody, startLine: 1 });

  // Skip splitting for very short documents
  if (boundedBody.length < MIN_DOC_CHARS_FOR_SPLIT) return fragments;

  const lines = boundedBody.split('\n');
  const remaining = () => MAX_FRAGMENTS_PER_DOC - fragments.length;

  // Extract sections (## headings)
  const sections = extractSections(lines);
  fragments.push(...sections.slice(0, remaining()));

  // Extract bullet lists
  if (remaining() > 0) {
    const lists = extractLists(lines);
    fragments.push(...lists.slice(0, remaining()));
  }

  // Extract code blocks
  if (remaining() > 0) {
    const blocks = extractCodeBlocks(lines);
    fragments.push(...blocks.slice(0, remaining()));
  }

  // Extract frontmatter facts
  if (frontmatter && remaining() > 0) {
    const fmFrags = extractFrontmatter(frontmatter);
    fragments.push(...fmFrags.slice(0, remaining()));
  }

  return fragments;
}

/**
 * Split observer-generated observations into fact and narrative fragments.
 * Used for documents that have structured `facts` and `narrative` fields.
 */
export function splitObservation(
  body: string,
  meta: { facts?: string; narrative?: string }
): Fragment[] {
  // Bound input size
  const boundedBody = body.length > MAX_SPLITTER_INPUT_CHARS
    ? body.slice(0, MAX_SPLITTER_INPUT_CHARS)
    : body;

  const fragments: Fragment[] = [];

  // Full document
  fragments.push({ type: 'full', label: null, content: boundedBody, startLine: 1 });

  // Individual facts
  if (meta.facts && fragments.length < MAX_FRAGMENTS_PER_DOC) {
    try {
      const facts = JSON.parse(meta.facts) as string[];
      for (const fact of facts) {
        if (fragments.length >= MAX_FRAGMENTS_PER_DOC) break;
        if (fact.length >= MIN_FRAGMENT_CHARS) {
          fragments.push({ type: 'fact', label: null, content: fact, startLine: 0 });
        }
      }
    } catch { /* invalid JSON, skip */ }
  }

  // Narrative
  if (meta.narrative && meta.narrative.length >= MIN_FRAGMENT_CHARS && fragments.length < MAX_FRAGMENTS_PER_DOC) {
    fragments.push({ type: 'narrative', label: null, content: meta.narrative, startLine: 0 });
  }

  return fragments;
}

// =============================================================================
// Section Extraction
// =============================================================================

function extractSections(lines: string[]): Fragment[] {
  const sections: Fragment[] = [];
  let currentHeading: string | null = null;
  let currentLines: string[] = [];
  let currentStartLine = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const headingMatch = line.match(/^(#{1,3})\s+(.+)/);

    if (headingMatch) {
      // Flush previous section
      if (currentHeading !== null && currentLines.length > 0) {
        const content = currentLines.join('\n').trim();
        if (content.length >= MIN_FRAGMENT_CHARS) {
          pushChunked(sections, 'section', currentHeading, content, currentStartLine);
        }
      }

      currentHeading = headingMatch[2]!.trim();
      currentLines = [line];
      currentStartLine = i + 1;
    } else {
      currentLines.push(line);
    }
  }

  // Flush last section
  if (currentHeading !== null && currentLines.length > 0) {
    const content = currentLines.join('\n').trim();
    if (content.length >= MIN_FRAGMENT_CHARS) {
      pushChunked(sections, 'section', currentHeading, content, currentStartLine);
    }
  }

  return sections;
}

// =============================================================================
// List Extraction
// =============================================================================

function extractLists(lines: string[]): Fragment[] {
  const lists: Fragment[] = [];
  let currentList: string[] = [];
  let listStartLine = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const isBullet = /^\s*[-*+]\s/.test(line) || /^\s*\d+\.\s/.test(line);
    // Indented continuation of a list item
    const isContinuation = currentList.length > 0 && /^\s{2,}/.test(line) && line.trim().length > 0;

    if (isBullet || isContinuation) {
      if (currentList.length === 0) listStartLine = i + 1;
      currentList.push(line);
    } else {
      if (currentList.length >= 2) {
        const content = currentList.join('\n').trim();
        if (content.length >= MIN_FRAGMENT_CHARS) {
          pushChunked(lists, 'list', null, content, listStartLine);
        }
      }
      currentList = [];
    }
  }

  // Flush trailing list
  if (currentList.length >= 2) {
    const content = currentList.join('\n').trim();
    if (content.length >= MIN_FRAGMENT_CHARS) {
      pushChunked(lists, 'list', null, content, listStartLine);
    }
  }

  return lists;
}

// =============================================================================
// Code Block Extraction
// =============================================================================

function extractCodeBlocks(lines: string[]): Fragment[] {
  const blocks: Fragment[] = [];
  let inBlock = false;
  let blockLines: string[] = [];
  let blockLang: string | null = null;
  let blockStartLine = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;

    if (!inBlock && line.match(/^```(\w*)/)) {
      inBlock = true;
      blockLang = line.match(/^```(\w+)/)?.[1] || null;
      blockLines = [line];
      blockStartLine = i + 1;
    } else if (inBlock && line.startsWith('```')) {
      blockLines.push(line);
      const content = blockLines.join('\n').trim();
      if (content.length >= MIN_FRAGMENT_CHARS) {
        pushChunked(blocks, 'code', blockLang, content, blockStartLine);
      }
      inBlock = false;
      blockLines = [];
      blockLang = null;
    } else if (inBlock) {
      blockLines.push(line);
    }
  }

  return blocks;
}

// =============================================================================
// Frontmatter Extraction
// =============================================================================

function extractFrontmatter(fm: Record<string, any>): Fragment[] {
  const fragments: Fragment[] = [];

  for (const [key, value] of Object.entries(fm)) {
    if (key === 'content_type' || key === 'tags') continue; // skip metadata-only fields

    let text: string;
    if (typeof value === 'string') {
      text = `${key}: ${value}`;
    } else if (typeof value === 'number' || typeof value === 'boolean') {
      text = `${key}: ${String(value)}`;
    } else if (Array.isArray(value)) {
      text = `${key}: ${value.join(', ')}`;
    } else {
      continue;
    }

    if (text.length >= 10) {
      fragments.push({
        type: 'frontmatter',
        label: key,
        content: text,
        startLine: 0,
      });
    }
  }

  return fragments;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Split content that exceeds MAX_FRAGMENT_CHARS into multiple chunks at
 * paragraph/line boundaries, with FRAGMENT_OVERLAP_CHARS of overlap between
 * consecutive chunks.
 *
 * Previously this truncated to a single MAX_FRAGMENT_CHARS chunk, which left
 * everything past the cap UNEMBEDDED — for large reference sections (e.g. a
 * frame-data movelist) most of the content had no vector at all, so vector
 * search could never surface it. Splitting keeps full coverage.
 */
function splitLargeContent(content: string): string[] {
  if (content.length <= MAX_FRAGMENT_CHARS) return [content];

  const chunks: string[] = [];
  let pos = 0;
  while (pos < content.length) {
    let end = Math.min(pos + MAX_FRAGMENT_CHARS, content.length);
    if (end < content.length) {
      // Prefer paragraph boundary, then line boundary, inside the window.
      const paragraphBreak = content.lastIndexOf('\n\n', end);
      if (paragraphBreak > pos + MAX_FRAGMENT_CHARS * 0.5) {
        end = paragraphBreak;
      } else {
        const lineBreak = content.lastIndexOf('\n', end);
        if (lineBreak > pos + MAX_FRAGMENT_CHARS * 0.5) {
          end = lineBreak;
        }
      }
    }
    const chunk = content.slice(pos, end).trim();
    if (chunk.length >= MIN_FRAGMENT_CHARS) chunks.push(chunk);
    if (end >= content.length) break;
    // Overlap so boundary-straddling facts appear whole in one chunk.
    pos = Math.max(pos + 1, end - FRAGMENT_OVERLAP_CHARS);
  }
  return chunks;
}

/**
 * Push one fragment per chunk of (possibly large) content. Multi-chunk
 * fragments get "(i/n)" suffixed labels; startLine is approximated for
 * continuation chunks by counting newlines.
 */
function pushChunked(
  out: Fragment[],
  type: Fragment['type'],
  label: string | null,
  content: string,
  startLine: number,
): void {
  const chunks = splitLargeContent(content);
  for (let i = 0; i < chunks.length; i++) {
    const chunkLabel = chunks.length > 1
      ? `${label ?? ''}${label ? ' ' : ''}(${i + 1}/${chunks.length})`.trim()
      : label;
    const offset = i === 0 ? 0 : countLines(content, content.indexOf(chunks[i]!));
    out.push({
      type,
      label: chunkLabel || null,
      content: chunks[i]!,
      startLine: startLine + offset,
    });
  }
}

function countLines(text: string, until: number): number {
  if (until <= 0) return 0;
  let n = 0;
  for (let i = 0; i < until && i < text.length; i++) {
    if (text.charCodeAt(i) === 10) n++;
  }
  return n;
}
