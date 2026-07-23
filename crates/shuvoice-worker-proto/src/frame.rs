//! Length-prefixed binary frames.

use std::io::{Read, Write};

use bytes::{BufMut, Bytes, BytesMut};
use uuid::Uuid;

use crate::error::ProtocolError;
use crate::limits::{
    BINARY_REQUEST_ID_LEN, MAX_FRAME_LEN, MAX_JSON_PAYLOAD_LEN, MIN_BINARY_PAYLOAD_LEN,
    MIN_FRAME_LEN,
};

/// Discriminator for the payload interpretation of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameKind {
    /// UTF-8 JSON control message ([`crate::messages::ControlMessage`]).
    Json = 1,
    /// Mono little-endian float32 PCM samples, prefixed by a request id.
    PcmF32Le = 2,
    /// Mono little-endian signed int16 PCM samples, prefixed by a request id.
    PcmI16Le = 3,
    /// Opaque bytes, prefixed by a request id.
    Bytes = 4,
}

impl FrameKind {
    /// Parse a raw kind byte.
    pub fn from_u8(value: u8) -> Result<Self, ProtocolError> {
        match value {
            1 => Ok(Self::Json),
            2 => Ok(Self::PcmF32Le),
            3 => Ok(Self::PcmI16Le),
            4 => Ok(Self::Bytes),
            other => Err(ProtocolError::UnsupportedFrameKind(other)),
        }
    }

    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn is_binary(self) -> bool {
        matches!(self, Self::PcmF32Le | Self::PcmI16Le | Self::Bytes)
    }
}

/// One length-prefixed protocol frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: FrameKind,
    pub payload: Bytes,
}

impl Frame {
    /// Construct a JSON control frame from already-serialized bytes.
    pub fn json_bytes(payload: impl Into<Bytes>) -> Result<Self, ProtocolError> {
        let payload = payload.into();
        check_json_payload_len(payload.len())?;
        check_payload_len(payload.len())?;
        Ok(Self {
            kind: FrameKind::Json,
            payload,
        })
    }

    /// Construct a binary frame with a leading request-id prefix.
    pub fn binary(kind: FrameKind, request_id: Uuid, body: &[u8]) -> Result<Self, ProtocolError> {
        if !kind.is_binary() {
            return Err(ProtocolError::InvalidBinaryPayload(
                "frame kind is not binary",
            ));
        }
        let total = BINARY_REQUEST_ID_LEN.checked_add(body.len()).ok_or(
            ProtocolError::InvalidBinaryPayload("payload length overflow"),
        )?;
        check_payload_len(total)?;

        let mut buf = BytesMut::with_capacity(total);
        buf.put_slice(request_id.as_bytes());
        buf.put_slice(body);
        Ok(Self {
            kind,
            payload: buf.freeze(),
        })
    }

    /// Convenience: float32 LE PCM frame.
    pub fn pcm_f32le(request_id: Uuid, samples: &[f32]) -> Result<Self, ProtocolError> {
        let mut body = BytesMut::with_capacity(samples.len().saturating_mul(4));
        for sample in samples {
            body.put_f32_le(*sample);
        }
        Self::binary(FrameKind::PcmF32Le, request_id, &body)
    }

    /// Convenience: int16 LE PCM frame.
    pub fn pcm_i16le(request_id: Uuid, samples: &[i16]) -> Result<Self, ProtocolError> {
        let mut body = BytesMut::with_capacity(samples.len().saturating_mul(2));
        for sample in samples {
            body.put_i16_le(*sample);
        }
        Self::binary(FrameKind::PcmI16Le, request_id, &body)
    }

    /// Total on-wire size including the 4-byte length prefix.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        4 + 1 + self.payload.len()
    }

    /// Encode into a newly allocated buffer.
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let mut out = BytesMut::with_capacity(self.encoded_len());
        self.encode_to(&mut out)?;
        Ok(out.freeze())
    }

    /// Encode into any bytes buffer.
    pub fn encode_to(&self, dst: &mut impl BufMut) -> Result<(), ProtocolError> {
        let length = encoded_length_field(self.payload.len())?;
        dst.put_u32(length);
        dst.put_u8(self.kind.as_u8());
        dst.put_slice(&self.payload);
        Ok(())
    }

    /// Encode to a std Write.
    pub fn write_to(&self, mut w: impl Write) -> Result<(), ProtocolError> {
        let bytes = self.encode()?;
        w.write_all(&bytes)?;
        Ok(())
    }

    /// Decode one frame from a full buffer (must contain exactly one frame, or leading frame).
    ///
    /// Returns `(frame, bytes_consumed)`.
    pub fn decode_from(buf: &[u8]) -> Result<(Self, usize), ProtocolError> {
        if buf.len() < 4 {
            return Err(ProtocolError::TruncatedFrame {
                declared: 4,
                got: buf.len(),
            });
        }
        let length = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        validate_length_field(length)?;
        let length_usize = length as usize;
        let total = 4 + length_usize;
        if buf.len() < total {
            return Err(ProtocolError::TruncatedFrame {
                declared: total,
                got: buf.len(),
            });
        }
        let kind = FrameKind::from_u8(buf[4])?;
        let payload = Bytes::copy_from_slice(&buf[5..total]);
        validate_kind_payload(kind, &payload)?;
        Ok((Self { kind, payload }, total))
    }

    /// Decode one frame from a std Read, allocating only after the length is validated.
    pub fn read_from(mut r: impl Read) -> Result<Self, ProtocolError> {
        let mut len_buf = [0u8; 4];
        read_exact_eof(&mut r, &mut len_buf, "frame length")?;
        let length = u32::from_be_bytes(len_buf);
        validate_length_field(length)?;

        // Allocate exactly `length` bytes for kind+payload — never attacker-controlled huge.
        let mut body = vec![0u8; length as usize];
        read_exact_eof(&mut r, &mut body, "frame body")?;
        let kind = FrameKind::from_u8(body[0])?;
        let payload = Bytes::copy_from_slice(&body[1..]);
        validate_kind_payload(kind, &payload)?;
        Ok(Self { kind, payload })
    }

    /// Split a binary payload into `(request_id, body)`.
    pub fn split_binary_payload(&self) -> Result<(Uuid, Bytes), ProtocolError> {
        if !self.kind.is_binary() {
            return Err(ProtocolError::InvalidBinaryPayload(
                "not a binary frame kind",
            ));
        }
        if self.payload.len() < MIN_BINARY_PAYLOAD_LEN {
            return Err(ProtocolError::InvalidBinaryPayload(
                "binary payload shorter than request id",
            ));
        }
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&self.payload[..BINARY_REQUEST_ID_LEN]);
        let id = Uuid::from_bytes(id_bytes);
        let body = self.payload.slice(BINARY_REQUEST_ID_LEN..);
        Ok((id, body))
    }

    /// Decode float32 LE samples from a PCM frame body (after request id strip).
    pub fn decode_f32le_samples(body: &[u8]) -> Result<Vec<f32>, ProtocolError> {
        if !body.len().is_multiple_of(4) {
            return Err(ProtocolError::InvalidBinaryPayload(
                "f32le PCM length not multiple of 4",
            ));
        }
        let mut out = Vec::with_capacity(body.len() / 4);
        for chunk in body.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    /// Decode int16 LE samples from a PCM frame body (after request id strip).
    pub fn decode_i16le_samples(body: &[u8]) -> Result<Vec<i16>, ProtocolError> {
        if !body.len().is_multiple_of(2) {
            return Err(ProtocolError::InvalidBinaryPayload(
                "i16le PCM length not multiple of 2",
            ));
        }
        let mut out = Vec::with_capacity(body.len() / 2);
        for chunk in body.chunks_exact(2) {
            out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(out)
    }
}

fn encoded_length_field(payload_len: usize) -> Result<u32, ProtocolError> {
    let length = payload_len
        .checked_add(1)
        .ok_or(ProtocolError::InvalidBinaryPayload("frame length overflow"))?;
    let length = u32::try_from(length).map_err(|_| ProtocolError::FrameTooLarge {
        length: u32::MAX,
        max: MAX_FRAME_LEN,
    })?;
    validate_length_field(length)?;
    Ok(length)
}

fn check_payload_len(payload_len: usize) -> Result<(), ProtocolError> {
    encoded_length_field(payload_len).map(|_| ())
}

fn check_json_payload_len(payload_len: usize) -> Result<(), ProtocolError> {
    let len = u32::try_from(payload_len).map_err(|_| ProtocolError::JsonTooLarge {
        length: u32::MAX,
        max: MAX_JSON_PAYLOAD_LEN,
    })?;
    if len > MAX_JSON_PAYLOAD_LEN {
        return Err(ProtocolError::JsonTooLarge {
            length: len,
            max: MAX_JSON_PAYLOAD_LEN,
        });
    }
    Ok(())
}

pub(crate) fn validate_kind_payload(kind: FrameKind, payload: &[u8]) -> Result<(), ProtocolError> {
    if kind == FrameKind::Json {
        check_json_payload_len(payload.len())?;
    }
    if kind.is_binary() && payload.len() < MIN_BINARY_PAYLOAD_LEN {
        return Err(ProtocolError::InvalidBinaryPayload(
            "binary payload shorter than request id",
        ));
    }
    Ok(())
}

pub(crate) fn validate_length_field(length: u32) -> Result<(), ProtocolError> {
    if length < MIN_FRAME_LEN {
        return Err(ProtocolError::FrameTooSmall {
            length,
            min: MIN_FRAME_LEN,
        });
    }
    if length > MAX_FRAME_LEN {
        return Err(ProtocolError::FrameTooLarge {
            length,
            max: MAX_FRAME_LEN,
        });
    }
    Ok(())
}

fn read_exact_eof(
    r: &mut impl Read,
    buf: &mut [u8],
    context: &'static str,
) -> Result<(), ProtocolError> {
    let mut read = 0;
    while read < buf.len() {
        match r.read(&mut buf[read..]) {
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
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(ProtocolError::Io(e)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrip_json_frame() {
        let frame = Frame::json_bytes(br#"{"type":"ping"}"#.as_slice()).unwrap();
        let encoded = frame.encode().unwrap();
        let (decoded, n) = Frame::decode_from(&encoded).unwrap();
        assert_eq!(n, encoded.len());
        assert_eq!(decoded, frame);
    }

    #[test]
    fn rejects_oversize_length_before_alloc_via_decode() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_LEN + 1).to_be_bytes());
        buf.push(1);
        let err = Frame::decode_from(&buf).unwrap_err();
        assert!(matches!(err, ProtocolError::FrameTooLarge { .. }));
    }

    #[test]
    fn read_from_rejects_oversize_without_reading_body() {
        let mut data = Vec::new();
        data.extend_from_slice(&(MAX_FRAME_LEN + 1).to_be_bytes());
        // No body supplied — must fail on length, not hang/alloc forever.
        let err = Frame::read_from(Cursor::new(data)).unwrap_err();
        assert!(matches!(err, ProtocolError::FrameTooLarge { .. }));
    }

    #[test]
    fn rejects_json_over_max_json_payload() {
        let huge = vec![b'x'; MAX_JSON_PAYLOAD_LEN as usize + 1];
        let err = Frame::json_bytes(huge).unwrap_err();
        assert!(matches!(err, ProtocolError::JsonTooLarge { .. }));
    }
}
