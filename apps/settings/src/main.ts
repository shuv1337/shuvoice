// Compiled entry: embed the patched addon and point the napi-rs loader at it
// before any @gpuix module loads (its createRequire lookup cannot be followed by
// `bun build --compile`, and the stock addon ignores physical input on
// multi-seat desktops; see patches/gpui-primary-seat.patch).
import addon from '../vendor/gpuix-native.linux-x64-gnu.node' with { type: 'file' }

process.env.NAPI_RS_NATIVE_LIBRARY_PATH ??= addon
await import('./start.tsx')
