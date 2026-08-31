import { Node, Edge } from '@xyflow/react';

export interface Folder {
  id: string;
  name: string;
  parentId: string | null;
  isExpanded: boolean;
}

export type ArchVariableType = 'int' | 'float' | 'bool' | 'string';

export type ArchVariableScope = 'init_param' | 'local_const';

export interface ArchVariable {
  id: string;               
  name: string;             
  type: ArchVariableType;   
  default: number | boolean | string | null;
  description?: string;
  scope: ArchVariableScope;
}

export interface GraphFile {
  id: string;
  name: string;
  parentId?: string | null;
  variables?: ArchVariable[];
  nodes: Node[];
  edges: Edge[];
  fileType?: 'graph' | 'code';
  compiledCode?: string;
}

export interface ArchIDEProject {
  name: string;
  version: string;
  entry_point: string;
  folders: Folder[];
  files: GraphFile[];
}

export type SidebarView = 'explorer' | 'library' | 'search';
