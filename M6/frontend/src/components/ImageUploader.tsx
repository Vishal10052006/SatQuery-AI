import { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, Sparkles, AlertCircle } from 'lucide-react';
import type { ViewMode } from '../types';

interface ImageSlotState {
  file?: File;
  previewUrl?: string;
  name: string;
}

interface ImageUploaderProps {
  mode: ViewMode;
  slot1: ImageSlotState;
  slot2: ImageSlotState;
  onChangeSlot1: (slot: ImageSlotState) => void;
  onChangeSlot2: (slot: ImageSlotState) => void;
  onLoadSample: () => void;
  sampleLabel?: string;
}

const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB limit

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  mode,
  slot1,
  slot2,
  onChangeSlot1,
  onChangeSlot2,
  onLoadSample,
  sampleLabel = 'Load Demo Satellite Imagery',
}) => {
  const isDualSlot = mode === 'change-detection' || mode === 'optical-sar';

  const fileInputRef1 = useRef<HTMLInputElement>(null);
  const fileInputRef2 = useRef<HTMLInputElement>(null);

  const [dragActive1, setDragActive1] = useState(false);
  const [dragActive2, setDragActive2] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const slot1Label = mode === 'optical-sar' ? 'Optical Imagery (Sentinel-2 / Landsat)' : mode === 'change-detection' ? 'Before Imagery (Timestamp T1)' : 'Primary Satellite Scene';
  const slot2Label = mode === 'optical-sar' ? 'SAR Imagery (Sentinel-1 / C-Band)' : 'After Imagery (Timestamp T2)';

  const validateAndProcessFile = (file: File, onDone: (slot: ImageSlotState) => void) => {
    setErrorMsg(null);
    if (!file.type.startsWith('image/') && !file.name.endsWith('.tiff') && !file.name.endsWith('.tif')) {
      setErrorMsg(`Invalid file type: "${file.name}". Please upload a standard satellite image (PNG, JPG, WebP, or TIFF).`);
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMsg(`File "${file.name}" exceeds the 25MB maximum limit.`);
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    onDone({
      file,
      previewUrl,
      name: file.name,
    });
  };

  const handleDrop = (e: React.DragEvent, slotIndex: 1 | 2) => {
    e.preventDefault();
    e.stopPropagation();
    if (slotIndex === 1) setDragActive1(false);
    else setDragActive2(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (slotIndex === 1) {
        validateAndProcessFile(file, onChangeSlot1);
      } else {
        validateAndProcessFile(file, onChangeSlot2);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent, slotIndex: 1 | 2) => {
    e.preventDefault();
    e.stopPropagation();
    if (slotIndex === 1) setDragActive1(true);
    else setDragActive2(true);
  };

  const handleDragLeave = (e: React.DragEvent, slotIndex: 1 | 2) => {
    e.preventDefault();
    e.stopPropagation();
    if (slotIndex === 1) setDragActive1(false);
    else setDragActive2(false);
  };

  return (
    <div style={{ marginBottom: 20 }}>
      {/* Upload Header with Sample Loader Button */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
            <ImageIcon size={18} color="var(--cyan)" />
            {isDualSlot ? 'Dual-Sensor / Bi-Temporal Image Inputs' : 'Satellite Scene Upload'}
          </h3>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Upload raw or orthorectified imagery (PNG, JPG, TIFF) or load curated SIH demo imagery.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onLoadSample}
          style={{ borderColor: 'var(--border-highlight)' }}
        >
          <Sparkles size={14} color="var(--cyan)" />
          {sampleLabel}
        </button>
      </div>

      {errorMsg && (
        <div className="alert alert-danger" style={{ marginBottom: 12 }}>
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>{errorMsg}</div>
        </div>
      )}

      {/* Grid of Upload Slots */}
      <div className="upload-grid" style={{ gridTemplateColumns: isDualSlot ? '1fr 1fr' : '1fr' }}>
        {/* Slot 1 */}
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--cyan)' }}>
              {slot1Label}
            </span>
            {slot1.previewUrl && (
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => onChangeSlot1({ name: '' })}
                style={{ color: 'var(--rose)', background: 'transparent', padding: '2px 6px' }}
                title="Remove image"
              >
                <X size={14} /> Clear
              </button>
            )}
          </div>

          {slot1.previewUrl ? (
            <div className="preview-container">
              <img src={slot1.previewUrl} alt="Slot 1 Preview" className="preview-img" />
              <div
                style={{
                  position: 'absolute',
                  bottom: 8,
                  left: 8,
                  background: 'rgba(0,0,0,0.7)',
                  padding: '3px 8px',
                  borderRadius: 4,
                  fontSize: '0.72rem',
                  color: '#fff',
                }}
              >
                {slot1.name || 'Satellite Scene'}
              </div>
            </div>
          ) : (
            <div
              className={`dropzone-box ${dragActive1 ? 'drag-over' : ''}`}
              onDragOver={(e) => handleDragOver(e, 1)}
              onDragLeave={(e) => handleDragLeave(e, 1)}
              onDrop={(e) => handleDrop(e, 1)}
              onClick={() => fileInputRef1.current?.click()}
            >
              <input
                ref={fileInputRef1}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/tiff"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    validateAndProcessFile(e.target.files[0], onChangeSlot1);
                  }
                }}
              />
              <div className="dropzone-icon">
                <UploadCloud size={24} />
              </div>
              <div style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: 4 }}>
                Drag & Drop or Click to Upload
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Supports PNG, JPG, GeoTIFF up to 25MB
              </div>
            </div>
          )}
        </div>

        {/* Slot 2 (if dual slot) */}
        {isDualSlot && (
          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: mode === 'change-detection' ? 'var(--emerald)' : 'var(--indigo)' }}>
                {slot2Label}
              </span>
              {slot2.previewUrl && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onChangeSlot2({ name: '' })}
                  style={{ color: 'var(--rose)', background: 'transparent', padding: '2px 6px' }}
                  title="Remove image"
                >
                  <X size={14} /> Clear
                </button>
              )}
            </div>

            {slot2.previewUrl ? (
              <div className="preview-container">
                <img src={slot2.previewUrl} alt="Slot 2 Preview" className="preview-img" />
                <div
                  style={{
                    position: 'absolute',
                    bottom: 8,
                    left: 8,
                    background: 'rgba(0,0,0,0.7)',
                    padding: '3px 8px',
                    borderRadius: 4,
                    fontSize: '0.72rem',
                    color: '#fff',
                  }}
                >
                  {slot2.name || 'Satellite Scene (T2 / SAR)'}
                </div>
              </div>
            ) : (
              <div
                className={`dropzone-box ${dragActive2 ? 'drag-over' : ''}`}
                onDragOver={(e) => handleDragOver(e, 2)}
                onDragLeave={(e) => handleDragLeave(e, 2)}
                onDrop={(e) => handleDrop(e, 2)}
                onClick={() => fileInputRef2.current?.click()}
              >
                <input
                  ref={fileInputRef2}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/tiff"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      validateAndProcessFile(e.target.files[0], onChangeSlot2);
                    }
                  }}
                />
                <div className="dropzone-icon" style={{ color: mode === 'change-detection' ? 'var(--emerald)' : 'var(--indigo)' }}>
                  <UploadCloud size={24} />
                </div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: 4 }}>
                  Drag & Drop or Click to Upload
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Supports PNG, JPG, GeoTIFF up to 25MB
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
