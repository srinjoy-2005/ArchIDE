/**
 * src/lib/utils.ts
 *
 * Shared pure utility functions used across the frontend.
 * Keeping these here avoids logic duplication between components
 * and makes them independently testable.
 */

import type { GraphFile, Folder } from './store';

/**
 * Resolves a GraphFile's full logical path relative to the 'graphs' root.
 *
 * Walks up the folder tree, skipping the sentinel 'graphs' root folder,
 * and appends the file's name without extension.
 *
 * Example: file "res_block.arch" inside folder "conv" inside "graphs"
 * returns "conv/res_block"
 *
 * This is the canonical source of truth for the file_id used in the
 * multi-graph compile/check payloads and VFS save calls.
 */
export function resolveFilePath(file: GraphFile, folders: Folder[]): string {
  const parts: string[] = [];
  let currFolderId = file.parentId ?? null;

  while (currFolderId) {
    const folder = folders.find((f) => f.id === currFolderId);
    if (!folder || folder.name === 'graphs') break;
    parts.unshift(folder.name);
    currFolderId = folder.parentId ?? null;
  }

  parts.push(file.name.replace(/\.[^/.]+$/, ''));
  return parts.join('/');
}
