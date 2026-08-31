export type ParamTypeName = 'int' | 'float' | 'bool' | 'string';

export interface ParamTypeHandler {
  inputType: 'number' | 'text' | 'checkbox';
  step?: number | string;
  isValid: (raw: string | number | boolean) => boolean;
  coerce: (raw: string | boolean) => number | boolean | string;
}

export const PARAM_TYPE_HANDLERS: Record<ParamTypeName, ParamTypeHandler> = {
  int: {
    inputType: 'number',
    step: 1,
    isValid: (v) => Number.isInteger(Number(v)) && !isNaN(Number(v)),
    coerce: (v) => parseInt(String(v), 10),
  },
  float: {
    inputType: 'number',
    step: 'any',
    isValid: (v) => !isNaN(parseFloat(String(v))),
    coerce: (v) => parseFloat(String(v)),
  },
  bool: {
    inputType: 'checkbox',
    isValid: () => true,
    coerce: (v) => v === 'true' || v === true,
  },
  string: {
    inputType: 'text',
    isValid: () => true,
    coerce: (v) => String(v),
  },
};

export function isValidIdentifier(name: string): boolean {
  return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name);
}
