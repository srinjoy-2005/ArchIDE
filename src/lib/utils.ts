/**
 * src/lib/utils.ts
 *
 * Shared pure utility functions used across the frontend.
 * Keeping these here avoids logic duplication between components
 * and makes them independently testable.
 */

import type { GraphFile, Folder } from './store';

/**
 * Resolves a GraphFile's full logical path relative to the graphs root.
 *
 * Walks up the folder tree, stopping at the graphsFolderId sentinel, and
 * appends the file's name without extension.
 *
 * Accepts graphsFolderId (stable UID) instead of the mutable folder name
 * so user renames don't break path resolution.
 *
 * Example: file "res_block.arch" inside folder "conv" inside "graphs"
 * returns "conv/res_block"
 *
 * This is the canonical source of truth for the file_id used in the
 * multi-graph compile/check payloads and VFS save calls.
 */
export function resolveFilePath(file: GraphFile, folders: Folder[], graphsFolderId?: string): string {
  const parts: string[] = [];
  let currFolderId = file.parentId ?? null;

  while (currFolderId) {
    const folder = folders.find((f) => f.id === currFolderId);
    // Stop at the graphs root folder (matched by stable UID if provided, else fall back to name)
    if (!folder || folder.id === graphsFolderId || folder.name === 'graphs') break;
    parts.unshift(folder.name);
    currFolderId = folder.parentId ?? null;
  }

  parts.push(file.name.replace(/\.[^/.]+$/, ''));
  return parts.join('/');
}
