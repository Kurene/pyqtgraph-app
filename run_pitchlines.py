# -*- coding: utf-8 -*-
from rasp_audio_stream import AudioInputStream
from pqg_pitchlines import PQGPitchLines


# PyAudioストリーム入力取得クラス
ais = AudioInputStream(CHUNK=4096) #, input_device_keyword="Real")
# メルスペクトログラム用クラス
pitchlines = PQGPitchLines( ais.RATE, (ais.CHANNELS, ais.CHUNK) )

# AudioInputStreamは別スレッドで動かす
import threading
thread = threading.Thread(target=ais.run, args=(pitchlines.callback_sigproc,))
thread.daemon=True
thread.start()

# スペクトログラム描画開始
pitchlines.run_app()