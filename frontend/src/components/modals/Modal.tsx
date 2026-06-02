import React from 'react'

interface Props {
  title: string
  onClose: () => void
  width?: string
  children: React.ReactNode
}

export default function Modal({ title, onClose, width = 'max-w-lg', children }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(30,30,26,0.45)' }}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className={`relative w-full ${width} rounded-sm border border-border shadow-2xl animate-modal-in flex flex-col max-h-[88dvh]`}
        style={{ background: 'var(--color-bg)' }}>
        {/* Modal header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border shrink-0">
          <h2 className="font-mono text-[11px] font-bold tracking-widest uppercase text-muted">{title}</h2>
          <button onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded-sm text-faint hover:text-muted hover:bg-surface2 transition-colors font-mono text-base">
            ×
          </button>
        </div>
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {children}
        </div>
      </div>
    </div>
  )
}
