// Brand images, embedded in the compiled binary. GPUI cannot read Bun's
// virtual filesystem, so they are handed to <img> as data URLs.
import splashPath from '../assets/splash.png' with { type: 'file' }
import lockupPath from '../assets/logo-lockup.png' with { type: 'file' }

export interface Brand {
  /** Lockup with tagline (the overlay splash art, edges faded to transparent). */
  splash: string
  /** Transparent lockup for the sidebar. */
  lockup: string
}

async function dataUrl(path: string): Promise<string> {
  const bytes = await Bun.file(path).bytes()
  return `data:image/png;base64,${Buffer.from(bytes).toString('base64')}`
}

export async function loadBrand(): Promise<Brand> {
  const [splash, lockup] = await Promise.all([dataUrl(splashPath), dataUrl(lockupPath)])
  return { splash, lockup }
}
