//! Async and sync framed readers/writers over arbitrary byte streams.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::error::ProtocolError;
use crate::frame::{Frame, FrameKind, validate_kind_payload, validate_length_field};
use crate::messages::ControlMessage;

/// Writes length-prefixed frames to an [`AsyncWrite`].
#[derive(Debug)]
pub struct FramedWriter<W> {
    inner: W,
}

impl<W> FramedWriter<W> {
    #[must_use]
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn into_inner(self) -> W {
        self.inner
    }

    #[must_use]
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    #[must_use]
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }
}

impl<W: AsyncWrite + Unpin> FramedWriter<W> {
    /// Write one frame and flush.
    pub async fn write_frame(&mut self, frame: &Frame) -> Result<(), ProtocolError> {
        let encoded = frame.encode()?;
        self.inner.write_all(&encoded).await?;
        self.inner.flush().await?;
        Ok(())
    }

    /// Serialize and write a JSON control message.
    pub async fn write_message(&mut self, message: &ControlMessage) -> Result<(), ProtocolError> {
        let frame = message.to_frame()?;
        self.write_frame(&frame).await
    }
}

/// Reads length-prefixed frames from an [`AsyncRead`].
///
/// Allocation is bounded: the length field is validated against
/// [`crate::limits::MAX_FRAME_LEN`] before any payload buffer is allocated.
#[derive(Debug)]
pub struct FramedReader<R> {
    inner: R,
}

impl<R> FramedReader<R> {
    #[must_use]
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }

    #[must_use]
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    #[must_use]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

impl<R: AsyncRead + Unpin> FramedReader<R> {
    /// Read exactly one frame from the stream.
    pub async fn read_frame(&mut self) -> Result<Frame, ProtocolError> {
        let mut len_buf = [0u8; 4];
        read_exact_async(&mut self.inner, &mut len_buf, "frame length").await?;
        let length = u32::from_be_bytes(len_buf);
        validate_length_field(length)?;

        let mut body = vec![0u8; length as usize];
        read_exact_async(&mut self.inner, &mut body, "frame body").await?;

        let kind = FrameKind::from_u8(body[0])?;
        let payload = bytes::Bytes::copy_from_slice(&body[1..]);
        validate_kind_payload(kind, &payload)?;
        Ok(Frame { kind, payload })
    }

    /// Read a frame and parse it as a JSON control message.
    pub async fn read_message(&mut self) -> Result<ControlMessage, ProtocolError> {
        let frame = self.read_frame().await?;
        ControlMessage::from_frame(&frame)
    }
}

/// Bidirectional framed connection (e.g. stdio pair or duplex stream).
#[derive(Debug)]
pub struct FramedConnection<R, W> {
    pub reader: FramedReader<R>,
    pub writer: FramedWriter<W>,
}

impl<R, W> FramedConnection<R, W> {
    #[must_use]
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader: FramedReader::new(reader),
            writer: FramedWriter::new(writer),
        }
    }

    #[must_use]
    pub fn into_inner(self) -> (R, W) {
        (self.reader.into_inner(), self.writer.into_inner())
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> FramedConnection<R, W> {
    pub async fn write_frame(&mut self, frame: &Frame) -> Result<(), ProtocolError> {
        self.writer.write_frame(frame).await
    }

    pub async fn write_message(&mut self, message: &ControlMessage) -> Result<(), ProtocolError> {
        self.writer.write_message(message).await
    }

    pub async fn read_frame(&mut self) -> Result<Frame, ProtocolError> {
        self.reader.read_frame().await
    }

    pub async fn read_message(&mut self) -> Result<ControlMessage, ProtocolError> {
        self.reader.read_message().await
    }
}

async fn read_exact_async<R: AsyncRead + Unpin>(
    r: &mut R,
    buf: &mut [u8],
    context: &'static str,
) -> Result<(), ProtocolError> {
    let mut read = 0usize;
    while read < buf.len() {
        match r.read(&mut buf[read..]).await {
            Ok(0) => {
                if read == 0 && context == "frame length" {
                    return Err(ProtocolError::UnexpectedEof { context });
                }
                return Err(ProtocolError::TruncatedFrame {
                    declared: buf.len(),
                    got: read,
                });
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(ProtocolError::Io(e)),
        }
    }
    Ok(())
}
