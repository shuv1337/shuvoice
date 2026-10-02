// Client for `shuvoice settings-bridge` (JSON lines on stdio, protocol v1).
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'

export const PROTOCOL_VERSION = 1

export interface Transport {
  send(line: string): void
  onLine(listener: (line: string) => void): void
  onClose(listener: (reason: string) => void): void
  close(): void
}

export class BridgeError extends Error {
  constructor(
    readonly kind: string,
    message: string,
    readonly data: Record<string, unknown> = {},
  ) {
    super(message)
  }
}

export interface BridgeEvent {
  event: string
  phase?: string
  busy?: string[]
  [key: string]: unknown
}

interface Pending {
  resolve: (value: unknown) => void
  reject: (error: BridgeError) => void
  onEvent?: (event: BridgeEvent) => void
}

export class Bridge {
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private closed: string | null = null

  constructor(private readonly transport: Transport) {
    transport.onLine((line) => this.receive(line))
    transport.onClose((reason) => {
      this.closed = reason
      for (const { reject } of this.pending.values()) {
        reject(new BridgeError('disconnected', reason))
      }
      this.pending.clear()
    })
  }

  /** `onEvent` receives progress events sent before the response (e.g. `apply`). */
  call<T>(op: string, params?: unknown, onEvent?: (event: BridgeEvent) => void): Promise<T> {
    if (this.closed !== null) {
      return Promise.reject(new BridgeError('disconnected', this.closed))
    }
    const id = this.nextId++
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onEvent })
      this.transport.send(JSON.stringify({ v: PROTOCOL_VERSION, id, op, ...(params === undefined ? {} : { params }) }))
    })
  }

  close() {
    this.transport.close()
  }

  private receive(line: string) {
    let message: {
      id?: number | null
      ok?: boolean
      event?: string
      result?: unknown
      error?: Record<string, unknown>
    }
    try {
      message = JSON.parse(line)
    } catch {
      return
    }
    if (typeof message.id !== 'number') return
    const pending = this.pending.get(message.id)
    if (!pending) return
    if (typeof message.event === 'string') {
      pending.onEvent?.(message as BridgeEvent)
      return
    }
    this.pending.delete(message.id)
    if (message.ok) {
      pending.resolve(message.result)
    } else {
      const { kind = 'unknown', message: text = 'bridge error', ...data } = message.error ?? {}
      pending.reject(new BridgeError(String(kind), String(text), data))
    }
  }
}

/** Locate `shuvoice` from the install layout; never from PATH. */
export function resolveShuvoiceBin(env: Record<string, string | undefined> = process.env): string {
  if (env.SHUVOICE_BIN) return env.SHUVOICE_BIN
  const sibling = join(dirname(process.execPath), 'shuvoice')
  if (existsSync(sibling)) return sibling
  return '/usr/bin/shuvoice'
}

/** Spawn the bridge as a child process and frame its stdout into lines. */
export function spawnBridge(bin: string, env = process.env): Transport {
  const child = Bun.spawn([bin, 'settings-bridge'], {
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'inherit',
    env,
  })
  const lineListeners: ((line: string) => void)[] = []
  const closeListeners: ((reason: string) => void)[] = []

  void (async () => {
    const decoder = new TextDecoder()
    let buffer = ''
    for await (const chunk of child.stdout) {
      buffer += decoder.decode(chunk, { stream: true })
      let newline = buffer.indexOf('\n')
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line) for (const listener of lineListeners) listener(line)
        newline = buffer.indexOf('\n')
      }
    }
    const code = await child.exited
    for (const listener of closeListeners) listener(`settings bridge exited (${code})`)
  })()

  return {
    send: (line) => {
      child.stdin.write(`${line}\n`)
      child.stdin.flush()
    },
    onLine: (listener) => lineListeners.push(listener),
    onClose: (listener) => closeListeners.push(listener),
    close: () => {
      child.stdin.end()
    },
  }
}
