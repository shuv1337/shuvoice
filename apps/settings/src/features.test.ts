import { expect, test } from 'bun:test'
import { diff, errorsByField, searchFields, supports } from './draft.ts'
import type { FieldMeta } from './schema.ts'

test('collections compare structurally, preserving order only for lists', () => {
  expect(diff({ list: ['One'], map: { a: 'A', b: 'B' }, optional: 'x' }, { list: ['One'], map: { b: 'B', a: 'A' }, optional: null })).toEqual({ optional: null })
  expect(diff({ list: ['A', 'B'] }, { list: ['B', 'A'] })).toEqual({ list: ['B', 'A'] })
  expect(diff({ map: { a: 'A' } }, { map: { a: '' } })).toEqual({ map: { a: '' } })
})

test('search spans ids, help and human section names', () => {
  const fields: FieldMeta[] = [{ id: 'tts.voice', section: 'text_to_speech', label: 'Voice', help: 'Speaker identity', unit: '', kind: { type: 'optional_text', max_len: 100 } }]
  for (const q of ['tts.voice', 'SPEAKER', 'text to speech', 'Text-to-Speech', 'voice identity']) expect(searchFields(fields, q)).toEqual(fields)
  expect(searchFields(fields, '   ')).toEqual([])
  expect(searchFields(fields, 'not present')).toEqual([])
})

test('old hello is fail-closed for optional operations', () => {
  expect(supports({}, 'model_download')).toBe(false)
  expect(supports(null, 'onboarding_defaults')).toBe(false)
  expect(supports({ features: ['models'] }, 'models')).toBe(true)
  expect(supports({ features: ['models'] }, 'model_download')).toBe(false)
})

test('Rust collection validation remains field-addressed', () => {
  expect(errorsByField([{ field: 'vocabulary.terms', message: 'Duplicate term' }, { field: 'typing.text_replacements', message: 'Empty key' }])).toEqual({ 'vocabulary.terms': 'Duplicate term', 'typing.text_replacements': 'Empty key' })
})
