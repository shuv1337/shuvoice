import QtQuick
import Quickshell
import Quickshell.Io
import qs.Ui

BarWidget {
  id: root
  moduleName: "shuv.shuvoice"
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  property string voiceState: "starting"
  property string glyph: "󰍬"
  property string details: "ShuVoice: starting"

  function refresh() {
    if (!status.running) status.running = true
  }

  function runAction(action) {
    Quickshell.execDetached(["shuvoice-waybar", action])
  }

  Process {
    id: status
    command: ["timeout", "5", "shuvoice-waybar", "status"]
    stdout: StdioCollector {
      onStreamFinished: {
        try {
          const result = JSON.parse(text)
          root.voiceState = result.alt || "error"
          root.glyph = result.text || "󰍭"
          root.details = (result.tooltip || "ShuVoice").replace("Right click: open action menu", "Right click: setup")
        } catch (e) {
          root.voiceState = "error"
          root.details = "ShuVoice: status unavailable"
        }
      }
    }
    onExited: function(code, exitStatus) {
      if (code !== 0) {
        root.voiceState = "error"
        root.details = "ShuVoice: status unavailable"
      }
    }
  }

  Timer {
    interval: 1000
    repeat: true
    running: true
    triggeredOnStart: true
    onTriggered: root.refresh()
  }

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.voiceState === "error" ? "" : root.glyph
    active: root.voiceState === "recording" || root.voiceState === "processing"
    tooltipText: root.details
    Accessible.name: "ShuVoice: " + root.voiceState
    onPressed: function(mouseButton) {
      if (mouseButton === Qt.MiddleButton) root.runAction("service-toggle")
      else if (mouseButton === Qt.RightButton) root.runAction("launch-wizard")
      else root.runAction("toggle-record")
    }
  }
}
