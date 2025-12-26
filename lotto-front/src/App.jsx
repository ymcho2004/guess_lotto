import { useState } from 'react';
import axios from 'axios';
import './App.css'; // CSS 파일 불러오기

function App() {
  const [lottoSets, setLottoSets] = useState([]); // 로또 번호 세트들 저장
  const [loading, setLoading] = useState(false);  // 로딩 상태

  // 백엔드한테 데이터 달라고 조르는 함수
  const fetchLottoNumbers = async () => {
    setLoading(true);
    try {
      // 우리가 만든 FastAPI 서버 주소
      const response = await axios.get('http://127.0.0.1:8000/predict?count=5');
      setLottoSets(response.data.predictions); // 받아온 데이터 저장
    } catch (error) {
      console.error("에러 났어요 ㅠㅠ", error);
      alert("백엔드 서버가 켜져 있는지 확인해주세요!");
    }
    setLoading(false);
  };

  // 공 색깔 정해주는 함수 (로또 공식 색상)
  const getBallColor = (num) => {
    if (num <= 10) return '#fbc400'; // 노랑
    if (num <= 20) return '#69c8f2'; // 파랑
    if (num <= 30) return '#ff7272'; // 빨강
    if (num <= 40) return '#aaaaaa'; // 회색
    return '#b0d840'; // 초록
  };

  return (
    <div className="container">
      <h1>🎰 AI 로또 예측기 🎰</h1>
      <p>LSTM Deep Learning Model Based</p>

      <button onClick={fetchLottoNumbers} disabled={loading}>
        {loading ? 'AI가 분석 중...' : '행운의 번호 5세트 받기 ✨'}
      </button>

      <div className="result-area">
        {lottoSets.map((set, index) => (
          <div key={index} className="lotto-set">
            <span className="set-label">{index + 1}세트</span>
            <div className="balls">
              {set.map((num, idx) => (
                <div 
                  key={idx} 
                  className="ball"
                  style={{ backgroundColor: getBallColor(num) }}
                >
                  {num}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;