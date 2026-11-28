use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_buffer::Buffer;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::{read_footer_length, FileDecoder};
use arrow_ipc::{root_as_footer, Block};
use arrow_schema::{ArrowError, SchemaRef};
use memmap2::Mmap;

use crate::types::{native_result, CacheError, CacheResult};

/// Read-only cache mapping; decoded Arrow buffers retain their mapped pages.
pub struct MappedMetadataReader {
    buffer: Buffer,
    schema: SchemaRef,
    decoder: FileDecoder,
    blocks: std::vec::IntoIter<Block>,
    version: arrow_ipc::MetadataVersion,
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
        let schema = native_result("metadata Arrow schema", || {
            Ok(Arc::new(fb_to_schema(
                footer.schema().ok_or_else(invalid_footer)?,
            )))
        })?;
        let mut decoder = FileDecoder::new(schema.clone(), footer.version());
        for block in footer.dictionaries().iter().flatten() {
            native_result("metadata Arrow dictionary", || {
                Ok(decoder.read_dictionary(block, &block_buffer(&buffer, block, footer_start)?)?)
            })?;
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
        let version = footer.version();
        Ok(Self {
            buffer,
            schema,
            decoder,
            blocks: blocks.into_iter(),
            version,
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
        let result = block_buffer(&self.buffer, &block, self.buffer.len())
            .map_err(|error| ArrowError::ParseError(error.to_string()))
            .and_then(|buffer| {
                native_result("metadata Arrow record batch", || {
                    Ok(self.decoder.read_record_batch(&block, &buffer)?)
                })
                .map_err(|error| ArrowError::ParseError(error.to_string()))
            })
            .and_then(|batch| {
                batch.ok_or_else(|| {
                    ArrowError::ParseError("missing metadata record batch".to_string())
                })
            });
        if result.is_err() {
            self.blocks = Vec::new().into_iter();
        }
        Some(result)
    }
}

impl RecordBatchReader for MappedMetadataReader {
    /// Return the validated schema without reading record batches.
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// A shared mapping indexes batch headers only; payload pages are faulted on demand.
#[derive(Debug)]
pub struct MappedMetadataIndex {
    buffer: Buffer,
    schema: SchemaRef,
    blocks: Vec<Block>,
    ends: Vec<usize>,
    version: arrow_ipc::MetadataVersion,
    validated: Vec<AtomicBool>,
}

impl MappedMetadataIndex {
    /// Index physical row ranges without decoding columns or retaining row objects.
    pub fn open(path: &Path, expected_rows: usize) -> CacheResult<Self> {
        let reader = MappedMetadataReader::open(path)?;
        let blocks = reader.blocks.collect::<Vec<_>>();
        let mut ends = Vec::with_capacity(blocks.len());
        let mut rows = 0_usize;
        for block in &blocks {
            let buffer = block_buffer(&reader.buffer, block, reader.buffer.len())?;
            let message = batch_message(&buffer, block)?;
            let batch = message
                .header_as_record_batch()
                .ok_or_else(invalid_footer)?;
            let count = usize::try_from(batch.length()).map_err(|_| invalid_footer())?;
            rows = rows.checked_add(count).ok_or_else(invalid_footer)?;
            ends.push(rows);
        }
        if rows != expected_rows {
            return Err(CacheError::InvalidCache(format!(
                "metadata row count does not match manifest: expected {expected_rows}, actual {rows}"
            )));
        }
        let validated = blocks.iter().map(|_| AtomicBool::new(false)).collect();
        Ok(Self {
            buffer: reader.buffer,
            schema: reader.schema,
            blocks,
            ends,
            version: reader.version,
            validated,
        })
    }

    /// Binary search handles byte-bounded, variable-sized and zero-row batches.
    pub fn location(&self, row: usize) -> CacheResult<(usize, usize)> {
        let index = self.ends.partition_point(|end| *end <= row);
        if index >= self.ends.len() {
            return Err(CacheError::InvalidCache(format!(
                "metadata row {row} is out of range"
            )));
        }
        let start = index
            .checked_sub(1)
            .and_then(|previous| self.ends.get(previous))
            .copied()
            .unwrap_or_default();
        Ok((index, row - start))
    }

    /// Decode only the addressed batch; primitive cache schemas need no dictionaries.
    pub fn batch(&self, index: usize) -> CacheResult<RecordBatch> {
        let block = self.blocks.get(index).ok_or_else(invalid_footer)?;
        let validated = self.validated.get(index).ok_or_else(invalid_footer)?;
        let buffer = block_buffer(&self.buffer, block, self.buffer.len())?;
        let batch = native_result("metadata Arrow random-access batch", || {
            let mut decoder = FileDecoder::new(self.schema.clone(), self.version);
            if validated.load(Ordering::Acquire) {
                // This exact block/schema has already passed the full Arrow decoder's
                // validation. Its mapping is immutable for every subsequent decode.
                decoder = unsafe { decoder.with_skip_validation(true) };
            }
            decoder
                .read_record_batch(block, &buffer)?
                .ok_or_else(invalid_footer)
        })?;
        validated.store(true, Ordering::Release);
        Ok(batch)
    }
}

/// Restrict flatbuffer parsing to the declared IPC message, excluding column bodies.
fn batch_message<'a>(buffer: &'a Buffer, block: &Block) -> CacheResult<arrow_ipc::Message<'a>> {
    let metadata_len = usize::try_from(block.metaDataLength()).map_err(|_| invalid_footer())?;
    let bytes = buffer
        .as_slice()
        .get(..metadata_len)
        .ok_or_else(invalid_footer)?;
    let prefix = if bytes.starts_with(&[255; 4]) { 8 } else { 4 };
    let message = bytes.get(prefix..).ok_or_else(invalid_footer)?;
    arrow_ipc::root_as_message(message).map_err(|error| {
        CacheError::InvalidCache(format!("invalid metadata batch header: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestDirectory;
    use arrow_array::{ArrayRef, StringArray};
    use arrow_ipc::writer::FileWriter;

    /// Signed block offsets and lengths cannot escape the mapped data section.
    #[test]
    fn malformed_block_ranges_are_rejected() -> CacheResult<()> {
        let buffer = Buffer::from(vec![0_u8; 64]);
        for block in [
            Block::new(-1, 0, 0),
            Block::new(8, -1, 0),
            Block::new(8, 0, -1),
            Block::new(0, 8, 0),
            Block::new(60, 8, 0),
            Block::new(i64::MAX, 8, i64::MAX),
        ] {
            assert!(block_buffer(&buffer, &block, 64).is_err());
        }
        assert_eq!(block_buffer(&buffer, &Block::new(8, 8, 16), 64)?.len(), 24);
        assert!(batch_message(&buffer, &Block::new(8, -1, 0)).is_err());
        assert!(batch_message(&buffer, &Block::new(8, 100, 0)).is_err());
        assert!(batch_message(&buffer, &Block::new(8, 3, 0)).is_err());
        let malformed = Buffer::from(vec![255_u8; 64]);
        assert!(batch_message(&malformed, &Block::new(8, 16, 0)).is_err());
        Ok(())
    }

    /// Bad file markers and footer lengths fail before a decoder can observe mapped slices.
    #[test]
    fn malformed_files_and_out_of_range_batches_are_rejected() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let path = directory.path().join("metadata.arrow");
        for bytes in [
            Vec::new(),
            b"not-arrow".to_vec(),
            b"ARROW1".to_vec(),
            b"ARROW1\0\0\xff\xff\xff\x7fARROW1".to_vec(),
        ] {
            std::fs::write(&path, bytes)?;
            assert!(MappedMetadataReader::open(&path).is_err());
        }
        write_metadata(&path)?;
        assert!(MappedMetadataIndex::open(&path, 2).is_err());
        let index = MappedMetadataIndex::open(&path, 1)?;
        assert!(index.location(1).is_err());
        assert!(index.batch(1).is_err());
        let mut reader = MappedMetadataReader::open(&path)?;
        reader.blocks = vec![Block::new(0, 0, 0)].into_iter();
        assert!(matches!(reader.next(), Some(Err(_))));
        assert!(reader.next().is_none());
        Ok(())
    }

    /// Dictionary IPC files decode values while their buffers retain the immutable mapping.
    #[test]
    fn dictionary_batches_round_trip_through_mapped_reader() -> CacheResult<()> {
        use arrow_array::types::Int32Type;
        use arrow_array::{DictionaryArray, Int32Array};
        let directory = TestDirectory::new()?;
        let path = directory.path().join("dictionary.arrow");
        let values = Arc::new(StringArray::from(vec!["first", "second"]));
        let array: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0]),
            values,
        )?);
        let batch = RecordBatch::try_from_iter([("dictionary", array)])?;
        let mut writer = FileWriter::try_new(File::create(&path)?, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
        let mut reader = MappedMetadataReader::open(&path)?;
        assert_eq!(reader.schema(), batch.schema());
        assert_eq!(reader.next().ok_or_else(invalid_footer)??, batch);
        assert!(reader.next().is_none());
        Ok(())
    }

    /// Create a native uncompressed string batch with a uniquely addressable UTF-8 value.
    fn write_metadata(path: &Path) -> CacheResult<()> {
        let array: ArrayRef = Arc::new(StringArray::from(vec!["unique-metadata-value"]));
        let batch = RecordBatch::try_from_iter([("text", array)])?;
        let mut writer = FileWriter::try_new(File::create(path)?, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    /// Successful validation belongs to the immutable mapping, not to any worker-local decoder.
    #[test]
    fn repeated_decoding_reuses_validated_buffers() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let path = directory.path().join("metadata.arrow");
        write_metadata(&path)?;
        let index = MappedMetadataIndex::open(&path, 1)?;
        let validated = index.validated.first().ok_or_else(invalid_footer)?;
        assert!(!validated.load(Ordering::Acquire));
        let first = index.batch(0)?;
        assert!(validated.load(Ordering::Acquire));
        assert_eq!(first, index.batch(0)?);
        Ok(())
    }

    /// Invalid UTF-8 must fail every attempt, never enabling the unchecked reuse path.
    #[test]
    fn failed_validation_never_marks_buffers_trusted() -> CacheResult<()> {
        let directory = TestDirectory::new()?;
        let path = directory.path().join("metadata.arrow");
        write_metadata(&path)?;
        let mut bytes = std::fs::read(&path)?;
        let offset = bytes
            .windows(b"unique-metadata-value".len())
            .position(|part| part == b"unique-metadata-value")
            .ok_or_else(invalid_footer)?;
        *bytes.get_mut(offset).ok_or_else(invalid_footer)? = 255;
        std::fs::write(&path, bytes)?;
        let index = MappedMetadataIndex::open(&path, 1)?;
        for _ in 0..2 {
            assert!(index.batch(0).is_err());
            assert!(!index
                .validated
                .first()
                .ok_or_else(invalid_footer)?
                .load(Ordering::Acquire));
        }
        Ok(())
    }
}
