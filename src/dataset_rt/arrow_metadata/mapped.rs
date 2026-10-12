use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_buffer::Buffer;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::{read_footer_length, FileDecoder};
use arrow_ipc::{root_as_footer, Block};
use arrow_schema::{ArrowError, SchemaRef};
use memmap2::Mmap;

use crate::types::{CacheError, CacheResult};

/// Read-only cache mapping; decoded Arrow buffers retain their mapped pages.
pub struct MappedMetadataReader {
    buffer: Buffer,
    schema: SchemaRef,
    decoder: FileDecoder,
    blocks: std::vec::IntoIter<Block>,
}

impl MappedMetadataReader {
    /// Map immutable metadata and validate footer and block bounds before decoding.
    pub fn open(path: &Path) -> CacheResult<Self> {
        let file = File::open(path)?;
        // Cache files are immutable while readers exist. Array owners retain the mapping
        // after this file handle closes and after buffers cross the Python boundary.
        let mapping = Arc::new(unsafe { Mmap::map(&file)? });
        let pointer = NonNull::new(mapping.as_ptr().cast_mut()).ok_or_else(invalid_footer)?;
        // This pointer spans mapping.len() bytes; the owner keeps that exact region alive.
        let buffer = unsafe { Buffer::from_custom_allocation(pointer, mapping.len(), mapping) };
        if !buffer.as_slice().starts_with(b"ARROW1") {
            return Err(invalid_footer());
        }
        let trailer_start = buffer.len().checked_sub(10).ok_or_else(invalid_footer)?;
        let trailer = buffer
            .as_slice()
            .get(trailer_start..)
            .ok_or_else(invalid_footer)?;
        let trailer = trailer.try_into().map_err(|_| invalid_footer())?;
        let footer_len = read_footer_length(trailer)?;
        let footer_start = trailer_start
            .checked_sub(footer_len)
            .ok_or_else(invalid_footer)?;
        let footer_bytes = buffer
            .as_slice()
            .get(footer_start..trailer_start)
            .ok_or_else(invalid_footer)?;
        let footer = root_as_footer(footer_bytes)
            .map_err(|error| CacheError::InvalidCache(error.to_string()))?;
        let schema = Arc::new(fb_to_schema(footer.schema().ok_or_else(invalid_footer)?));
        let mut decoder = FileDecoder::new(schema.clone(), footer.version());
        for block in footer.dictionaries().iter().flatten() {
            decoder.read_dictionary(block, &block_buffer(&buffer, block, footer_start)?)?;
        }
        let blocks = footer
            .recordBatches()
            .ok_or_else(invalid_footer)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for block in &blocks {
            block_buffer(&buffer, block, footer_start)?;
        }
        Ok(Self {
            buffer,
            schema,
            decoder,
            blocks: blocks.into_iter(),
        })
    }
}

/// Reject malformed footer locations before slicing mapped memory.
fn invalid_footer() -> CacheError {
    CacheError::InvalidCache("invalid metadata Arrow footer".to_string())
}

/// Validate signed lengths and file bounds before giving Arrow a shared block view.
fn block_buffer(buffer: &Buffer, block: &Block, data_end: usize) -> CacheResult<Buffer> {
    let offset = usize::try_from(block.offset()).map_err(|_| invalid_footer())?;
    let metadata_len = usize::try_from(block.metaDataLength()).map_err(|_| invalid_footer())?;
    let body_len = usize::try_from(block.bodyLength()).map_err(|_| invalid_footer())?;
    let len = metadata_len
        .checked_add(body_len)
        .ok_or_else(invalid_footer)?;
    let end = offset.checked_add(len).ok_or_else(invalid_footer)?;
    if offset < 8 || end > data_end {
        return Err(invalid_footer());
    }
    Ok(buffer.slice_with_length(offset, len))
}

impl Iterator for MappedMetadataReader {
    type Item = Result<RecordBatch, ArrowError>;

    /// Decode one batch; properly aligned uncompressed buffers remain shared.
    fn next(&mut self) -> Option<Self::Item> {
        let block = self.blocks.next()?;
        Some(
            block_buffer(&self.buffer, &block, self.buffer.len())
                .map_err(|error| ArrowError::ParseError(error.to_string()))
                .and_then(|buffer| self.decoder.read_record_batch(&block, &buffer))
                .and_then(|batch| {
                    batch.ok_or_else(|| {
                        ArrowError::ParseError("missing metadata record batch".to_string())
                    })
                }),
        )
    }
}

impl RecordBatchReader for MappedMetadataReader {
    /// Return the validated schema without reading record batches.
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
