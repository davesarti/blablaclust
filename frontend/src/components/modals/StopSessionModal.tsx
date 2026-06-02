import React, { useState } from 'react'
import { useApp } from '../../store/AppContext'
import { patchSessionState } from '../../api/client'
import Modal from './Modal'

interface Props { onClose: () => void; onClosed: () => void }

export default function StopSessionModal({ onClose, onClosed }: Props) {
  const { state, dispatch } = useApp()
  const sess = state.session!
  const [loading, setLoading] = useState(false)

  function exportJSON() {
    const data = { session: sess.session, clusters: sess.clusters }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${sess.sessionId}_clusters.json`
    a.click()
  }

  function exportCSV() {
    const header = 'id,name,description,size'
    const rows = sess.clusters.map(c =>
      [c.id, `"${c.name.replace(/"/g, '""')}"`, `"${c.description.replace(/"/g, '""')}"`, c.size].join(',')
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${sess.sessionId}_clusters.csv`
    a.click()
  }

  async function closeSession() {
    setLoading(true)
    try {
      await patchSessionState(sess.sessionId, 'closed')
      dispatch({ type: 'UPDATE_STATUS', status: 'closed' })
      onClosed()
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal title="Stop session" onClose={onClose}>
      <div className="p-5 flex flex-col gap-4">
        <p className="text-[14px] text-muted leading-relaxed">
          Export your clusters before closing, or close without saving.
        </p>

        <div className="flex flex-col gap-2">
          <button onClick={exportJSON}
            className="w-full py-2.5 rounded-sm border border-border text-[15px] font-medium text-muted hover:bg-surface2 transition-colors text-left px-4">
            ↓ Export JSON
          </button>
          <button onClick={exportCSV}
            className="w-full py-2.5 rounded-sm border border-border text-[15px] font-medium text-muted hover:bg-surface2 transition-colors text-left px-4">
            ↓ Export CSV
          </button>
        </div>

        <div className="h-px bg-border" />

        <button onClick={closeSession} disabled={loading}
          className="w-full py-2.5 rounded-sm border border-red-200 text-[15px] font-medium text-red-700 bg-red-50 hover:bg-red-100 transition-colors disabled:opacity-50">
          {loading ? 'Closing…' : 'Close without exporting'}
        </button>
      </div>
    </Modal>
  )
}
