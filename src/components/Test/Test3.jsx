import React, { useState } from 'react';





import './test.scss';
const questions = [
  {
    title: '1.	Графикалық примитив дегеніміз:',
    variants: ["a)	растрлық графикалық редактор құралы","b)	бейне жадындағы суреттің бір пикселінің сипаттамасы","c)	векторлық графикалық кескінді қалыптастырудағы қарапайым элемент","d)	Paint графикалық редакторының көмегімен жасалған қарапайым сурет."],
    correct: 2,
  },
  {
    title: '2.	AutoCAD-та осы құралдар тақтасының командалары не үшін қажет?',
    variants: ["a)	нысандарды сызу үшін","b)	нысандарды өңдеу үшін","c)	қабаттарды құру үшін","d)	қабаттардың қасиеттерін өңдеу үшін."],
    correct: 2,
  },
  {
    title: '3.	AutoCAD-та белгіленген аралықта осындай объектілерді құру үшін қандай команда қолданылады?',
    variants: ["a)","b)","c)","d)"],
    correct: 2,
  },
  {
    title: '4.	Қандай құрал қабат элементтерін көрінбейтін етеді?',
    variants: ["a)","b)","c)","d)"],
    correct: 2,
  },
  {
    title: '5.	Dimline опциясы және автокад мүмкіндік береді:',
    variants: ["a) End point","b) Midpoint","c) Center","d) Node"],
    correct: 2,
  },
  {
    title: '6.	AutoCad файлдары қандай кеңейтілікте болады:',
    variants: ["a) .doc","b) .dwg","c) .acad","d) .pdf"],
    correct: 2,
  },
  {
    title: '7. AutoCad –өңдеушісі кім?',
    variants: ["a)	Apple","b)	AutoDesk","c)	Unix","d)	Adobe"],
    correct: 2,
  },
  {
    title: '8. AutoCad –өңдеушісі кім?',
    variants: ["a)	Apple","b)	AutoDesk","c)	Unix","d)	Microsoft"],
    correct: 2,
  },
  {
    title: '9. Соңғы команданы қайталау үшін қай түймені басу қажет?',
    variants: ["a)	 Ғ1","b)	 Enter","c)	 Esc","d)	 F2"],
    correct: 2,
  },
  {
    title: '10. Линия командасының кілттері:',
    variants: ["a)	 Close, Undo","b)	 Hor, Ver, Ang","c)	 Angle, Center, Radius","d)	 Length, Undo, Width"],
    correct: 2,
  },
  {
    title: '11. Басы, соңы және бастапқы нүктесінен жүргізілген жанаманың көлбеулік бұрышы бойынша доғаны қай тәсілмен сызады?',
    variants: ["a)	 3Point","b)	 Arccont","c)	 St, End, Dir","d)	 St, End, Rad"],
    correct: 2,
  },
  {
    title: '12. Жиналмалы меню командасына жатпайтындар:',
    variants: ["a)	 Файл","b)	 Формат","c)	 Размер","d)	 Вид"],
    correct: 2,
  },
  {
    title: '13. Доға режимінде радиус енгізу кілтін көрсетіңіз.',
    variants: ["a)	 Undo","b)	 Angle","c)	 Line","d)	 Radius"],
    correct: 2,
  },
  {
    title: '14. Кесінді салуда нүктенің салыстырмалы полярлық координаталарын қалай енгізеді?',
    variants: ["a)	 х, у","b)	 @ dx, dy","c)	 x, y, z","d)	 @ r, <A"],
    correct: 2,
  },
  {
    title: '15. Дуга режимінде ChordLength  кілті қандай қызмет атқарады?',
    variants: ["a)	 хорданың ұзындығын енгізу","b)	 Доғаның центрлік бұрышын енгізу","c)	 Доғаның центрін енгізу","d)	 Доғаның жарты қалыңдығын енгізу"],
    correct: 2,
  },
  {
    title: '16. Полигон командасы қай саймандар панелінде орналасқан?',
    variants: ["a)	 Рисовать","b)	 Вставить","c)	 Формат","d)	 Изменить"],
    correct: 2,
  },
  {
    title: '17. Erase командасын пайдаланбай объектіні қай батырманың көмегімен өшіруге болады?',
    variants: ["a)	Insert ","b)	Home","c)	End"],
    correct: 2,
  },
  {
    title: '18. Мәтіндік және графикалық терезелерді ауыстыру пернесі',
    variants: ["a)	F1 ","b)	F2","c)	F3","d)	F4"],
    correct: 2,
  },
  {
    title: '19. Линия командасының кілттері',
    variants: ["a)	Hor, Ver, Ang, Bisect, Offset ","b)	Angle, Center, Close","c)	Arc, Width, Undo","d)	Line, Width"],
    correct: 2,
  },
  {
    title: '20. Графикалық алаңшаның түсін қалай өзгертеміз?',
    variants: ["a)	Tools = > Options = > screen = > Color = > Color selection","b)	Tools = > Color selection","c)	Tools = > Color ","d)	Draw => Line = > Color selection "],
    correct: 2,
  },
  {
    title: '21. Берілген нүктеден өтетін горизанталь түзу сызу үшін LINE командасының қай кілті қажет?',
    variants: ["a)	Ver ","b)	Hor","c)	Ang ","d)	Bisect"],
    correct: 2,
  },
  {
    title: '22. Менюдің SIZE элементі не үшін қолданылады?',
    variants: ["a)	сызба өлшемдерін қою ","b)	блоктарды және объектілерді қою ","c)	қабаттармен , түстермен сызықтар типтерімени жұмыс  ","d)	сызу командалары "],
    correct: 2,
  },
  // {
  //   title: '',
  //   variants: ["","","",""],
  //   correct: 2,
  // }
];

function Result( {correct}) {
  return (
    <div className="result">
      <img src="https://cdn-icons-png.flaticon.com/512/2278/2278992.png" />
      <h2>Сіз {questions.length} сұрақтын {correct} сұрағына дұрыс жауап бердіңіз </h2>
      <a href="/test"> <button>Тестті аяқтау</button></a>
    </div>
  );
}

function Game({question, onClickVariants, step}) {
  return (
    <>
      <div className="progress">
        <div style={{ width: `${Math.round((step / questions.length) * 100)}%` }} className="progress__inner"></div>
      </div>
      <h1>{question.title}</h1>
      <ul>
        {
          question.variants.map((text, index) => 
            (<li onClick={() => onClickVariants(index)} key={text}>{text}</li>)
          )
        }
          
      </ul>
    </>
  );
}

function Test3() {
  const [step, setStep] = useState(0);
  const [correct, setCorrect] = useState(0);
  const question = questions[step]
  const onClickVariants = (index) => {
   setStep( step + 1)

   if(index === question.correct) {
    setCorrect(correct + 1)
   }
  }
  return (
    <div className="test">
      {
       step !== questions.length ? <Game step={step} question={question} onClickVariants={onClickVariants} /> :
      (<Result correct={correct} />)
      }
    </div>
  );
}

export default Test3