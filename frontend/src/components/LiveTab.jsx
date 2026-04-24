import { useRef, useState, useEffect, useCallback } from 'react'
import { api } from '../api'

const EMOTION_COLOR = {
  기쁨: '#f59e0b', 당황: '#f97316', 분노: '#ef4444', 상처: '#6366f1',
  불안: '#8b5cf6', 슬픔: '#3b82f6', 중립: '#71717a',
}
const EMOTION_EMOJI = {
  기쁨: '😄', 당황: '😳', 분노: '😡', 상처: '😢',
  불안: '😰', 슬픔: '😢', 중립: '😐',
}

const MODELS = [
  { id: 'densenet121',     label: 'DenseNet121 (87.6%)' },
  { id: 'densenet121_new', label: 'DenseNet121 재학습 (83.8%)' },
]

const INTERVAL_MS = 1000  // 분석 주기 (ms)

function drawOverlay(canvas, result) {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!result) return

  const { emotion, confidence, scores } = result
  const color = EMOTION_COLOR[emotion] || '#6366f1'
  const emoji = EMOTION_EMOJI[emotion] || '🤔'

  const W = canvas.width
  const H = canvas.height

  // 반투명 상단 배너
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.beginPath()
  ctx.roundRect(12, 12, 200, 72, 12)
  ctx.fill()

  // 이모지
  ctx.font = '32px serif'
  ctx.fillText(emoji, 22, 52)

  // 감정 텍스트
  ctx.fillStyle = color
  ctx.font = 'bold 20px -apple-system, sans-serif'
  ctx.fillText(emotion, 62, 42)

  // 신뢰도 텍스트
  ctx.fillStyle = '#e4e4e7'
  ctx.font = '13px -apple-system, sans-serif'
  ctx.fillText(`${(confidence * 100).toFixed(1)}%`, 62, 62)

  // 하단 신뢰도 바 (4감정)
  const barW = W - 24
  const barH = 6
  const barY = H - 60
  const emotions = Object.entries(scores || {})
  const colW = barW / Math.max(emotions.length, 1)

  emotions.forEach(([em, score], i) => {
    const x = 12 + i * colW
    const emColor = EMOTION_COLOR[em] || '#6366f1'

    // 배경
    ctx.fillStyle = 'rgba(0,0,0,0.45)'
    ctx.beginPath()
    ctx.roundRect(x, barY, colW - 4, barH + 24, 4)
    ctx.fill()

    // 바
    ctx.fillStyle = em === emotion ? emColor : emColor + '77'
    ctx.beginPath()
    ctx.roundRect(x, barY, (colW - 4) * score, barH, 3)
    ctx.fill()

    // 라벨
    ctx.fillStyle = em === emotion ? emColor : '#a1a1aa'
    ctx.font = `${em === emotion ? 'bold' : 'normal'} 11px -apple-system, sans-serif`
    ctx.textAlign = 'center'
    ctx.fillText(em, x + (colW - 4) / 2, barY + barH + 14)
    ctx.textAlign = 'left'
  })
}

export default function LiveTab() {
  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)
  const streamRef   = useRef(null)
  const intervalRef = useRef(null)
  const pendingRef  = useRef(false)

  const [active, setActive]     = useState(false)
  const [result, setResult]     = useState(null)
  const [fps, setFps]           = useState(0)
  const [model, setModel]       = useState('densenet121')
  const [error, setError]       = useState(null)

  const lastFpsRef = useRef(Date.now())
  const fpsCountRef = useRef(0)

  // 캔버스 크기를 video에 맞춤
  const syncCanvasSize = useCallback(() => {
    const v = videoRef.current
    const c = canvasRef.current
    if (!v || !c) return
    c.width  = v.videoWidth  || v.clientWidth
    c.height = v.videoHeight || v.clientHeight
  }, [])

  // 프레임 캡처 → API 호출 → 오버레이
  const captureAndAnalyze = useCallback(async () => {
    if (pendingRef.current) return
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    pendingRef.current = true
    try {
      const cap = document.createElement('canvas')
      cap.width  = video.videoWidth
      cap.height = video.videoHeight
      cap.getContext('2d').drawImage(video, 0, 0)
      const b64 = cap.toDataURL('image/jpeg', 0.8).split(',')[1]

      const res = await api.analyzeBase64(b64, model)
      const data = res.data
      setResult(data)
      syncCanvasSize()
      drawOverlay(canvas, data)

      // FPS 계산
      fpsCountRef.current += 1
      const now = Date.now()
      if (now - lastFpsRef.current >= 3000) {
        setFps(Math.round(fpsCountRef.current / ((now - lastFpsRef.current) / 1000)))
        fpsCountRef.current = 0
        lastFpsRef.current  = now
      }
    } catch (e) {
      console.error('[live]', e)
    } finally {
      pendingRef.current = false
    }
  }, [model, syncCanvasSize])

  const startLive = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      const v = videoRef.current
      v.srcObject = stream
      await v.play()
      setActive(true)
      intervalRef.current = setInterval(captureAndAnalyze, INTERVAL_MS)
    } catch {
      setError('카메라 접근 권한이 필요합니다.')
    }
  }, [captureAndAnalyze])

  const stopLive = useCallback(() => {
    clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    const c = canvasRef.current
    if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height)
    setActive(false)
    setResult(null)
    setFps(0)
  }, [])

  // 탭 언마운트 시 정리
  useEffect(() => () => stopLive(), [stopLive])

  // 모델 변경 시 인터벌 재시작
  useEffect(() => {
    if (!active) return
    clearInterval(intervalRef.current)
    intervalRef.current = setInterval(captureAndAnalyze, INTERVAL_MS)
  }, [model, active, captureAndAnalyze])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* 모델 선택 */}
      <div className="section" style={{ paddingBottom: 8 }}>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          disabled={active}
          style={{
            width: '100%', padding: '10px 12px', borderRadius: 10,
            border: '1.5px solid var(--border)', fontSize: 14,
            background: 'var(--surface)', color: 'var(--text)',
            appearance: 'none',
          }}
        >
          {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>

      {/* 비디오 + 캔버스 오버레이 */}
      <div style={{ position: 'relative', flex: 1, background: '#000', margin: '0 16px', borderRadius: 12, overflow: 'hidden', minHeight: 280 }}>
        <video
          ref={videoRef}
          autoPlay playsInline muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: 'block' }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'scaleX(-1)', pointerEvents: 'none' }}
        />
        {!active && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', flexDirection: 'column', gap: 8,
          }}>
            <span style={{ fontSize: 48 }}>📷</span>
            <span style={{ color: '#a1a1aa', fontSize: 14 }}>카메라를 시작하세요</span>
          </div>
        )}
        {active && fps > 0 && (
          <div style={{
            position: 'absolute', top: 12, right: 12,
            background: 'rgba(0,0,0,0.5)', color: '#a1a1aa',
            fontSize: 11, padding: '3px 7px', borderRadius: 6,
          }}>
            {fps} fps
          </div>
        )}
      </div>

      {/* 현재 결과 요약 */}
      {result && (
        <div style={{
          margin: '10px 16px 0', padding: '10px 14px',
          background: 'var(--bg)', borderRadius: 10,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <span style={{ fontSize: 28 }}>{EMOTION_EMOJI[result.emotion] || '🤔'}</span>
          <div>
            <div style={{ fontWeight: 700, color: EMOTION_COLOR[result.emotion] || '#6366f1' }}>
              {result.emotion}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              신뢰도 {(result.confidence * 100).toFixed(1)}% · {result.infer_ms}ms
            </div>
          </div>
        </div>
      )}

      {/* 에러 */}
      {error && (
        <div className="section" style={{ paddingTop: 8 }}>
          <div className="notice">{error}</div>
        </div>
      )}

      {/* 컨트롤 버튼 */}
      <div className="section" style={{ paddingTop: 10 }}>
        {!active ? (
          <button className="btn btn-primary btn-full" onClick={startLive}>
            🎥 실시간 분석 시작
          </button>
        ) : (
          <button className="btn btn-outline btn-full" onClick={stopLive}>
            ⏹ 중지
          </button>
        )}
      </div>

    </div>
  )
}
