import { describe, expect, test } from 'bun:test'
import { Bridge, BridgeError, type Transport, resolveShuvoiceBin } from './bridge.ts'

function fakeTransport() {
  const sent: Record<string, unknown>[] = []
  let line: (l: string) => void = () => {}
  let close: (r: string) => void = () => {}
  const transport: Transport = {
    send: (l) => sent.push(JSON.parse(l)),
    onLine: (listener) => (line = listener),
    onClose: (listener) => (close = listener),
    close: () => {},
  }
  return { transport, sent, reply: (msg: unknown) => line(JSON.stringify(msg)), close: (r: string) => close(r) }
}

describe('Bridge', () => {
  test('correlates responses by id, out of order', async () => {
    const t = fakeTransport()
    const bridge = new Bridge(t.transport)
    const a = bridge.call<string>('hello')
    const b = bridge.call<string>('status', { x: 1 })
    expect(t.sent).toEqual([
      { v: 1, id: 1, op: 'hello' },
      { v: 1, id: 2, op: 'status', params: { x: 1 } },
    ])
    t.reply({ v: 1, id: 2, ok: true, result: 'B' })
    t.reply('not json'.length)
    t.reply({ v: 1, id: 1, ok: true, result: 'A' })
    expect(await a).toBe('A')
    expect(await b).toBe('B')
  })

  test('maps error responses to BridgeError with kind and data', async () => {
    const t = fakeTransport()
    const bridge = new Bridge(t.transport)
    const call = bridge.call('save', {})
    t.reply({ v: 1, id: 1, ok: false, error: { kind: 'conflict', message: 'changed', current_revision: 'abc' } })
    const error = (await call.catch((e) => e)) as BridgeError
    expect(error).toBeInstanceOf(BridgeError)
    expect(error.kind).toBe('conflict')
    expect(error.data.current_revision).toBe('abc')
  })

  test('rejects pending and later calls after the bridge exits', async () => {
    const t = fakeTransport()
    const bridge = new Bridge(t.transport)
    const pending = bridge.call('snapshot')
    t.close('settings bridge exited (1)')
    expect(((await pending.catch((e) => e)) as BridgeError).kind).toBe('disconnected')
    expect(((await bridge.call('hello').catch((e) => e)) as BridgeError).kind).toBe('disconnected')
  })

  test('resolves the shuvoice binary from env or install layout, never PATH', () => {
    expect(resolveShuvoiceBin({ SHUVOICE_BIN: '/opt/sv/shuvoice' })).toBe('/opt/sv/shuvoice')
    const fallback = resolveShuvoiceBin({ PATH: '/somewhere/else' })
    expect(fallback.endsWith('/shuvoice')).toBe(true)
    expect(fallback.startsWith('/somewhere')).toBe(false)
  })
})

describe('Bridge events', () => {
  test('routes progress events to the call, then resolves', async () => {
    const t = fakeTransport()
    const bridge = new Bridge(t.transport)
    const phases: string[] = []
    const call = bridge.call('apply', { changes: {} }, (e) => phases.push(String(e.phase)))
    t.reply({ v: 1, id: 1, event: 'progress', phase: 'saving' })
    t.reply({ v: 1, id: 1, event: 'progress', phase: 'restarting' })
    t.reply({ v: 1, id: 1, ok: true, result: { restart: { outcome: 'ready' } } })
    expect(await call).toEqual({ restart: { outcome: 'ready' } })
    expect(phases).toEqual(['saving', 'restarting'])
  })
})
