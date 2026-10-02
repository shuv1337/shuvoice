import { motion } from '@gpuix/react'
import { C } from '../theme.ts'
import { Text } from './controls.tsx'

/** Matches the edges of the splash art so it blends into the window. */
const SPLASH_BG = '#050208'
/** Keep the splash up at least this long so it reads as intentional, not a flash. */
export const SPLASH_MIN_MS = 1000

export function Splash({ src }: { src: string }) {
  return (
    <motion.div
      initial={{ opacity: 1 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: SPLASH_BG,
      }}
    >
      <div testId="splash" style={{ width: '80%', height: '50%', maxWidth: 760, maxHeight: 362 }}>
        <img
          src={src}
          alt="ShuVoice"
          objectFit="contain"
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      <Text color={C.dim}>Loading settings…</Text>
    </motion.div>
  )
}
