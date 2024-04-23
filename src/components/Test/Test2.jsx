import React, { useState } from 'react';





import './test.scss';
const questions = [
  {
    title: '1.	AutoCAD редакторлардың қай түріне жатады?',
    variants: ['a)	Растрлық',
      'b)	Мәтіндік. ',
      'c)	Векторлық',
      'd)	Кестелік'
      ],
    correct: 2,
  },
  {
    title: '2.	AutoCAD-қа полярлық координаттарды енгізу үшін қандай таңба қолданылады?',
    variants: ["a)	<","b)	>","c)	@", "d)	="],
    correct: 2,
  },
  {
    title: '3.	Командалардың қайсысы AutoCAD ішіндегі нысанның өлшемін өзгертпейді?',
    variants: ["a)","b)","c)","d)"],
    correct: 2,
  },
  {
    title: '4.	AutoCAD-та төмендегі құралдар тақтасының командалары не үшін арналған?',
    variants: ["a)	Өлшемдерді орналастыру үшін","b)	нысандарды өңдеу үшін","c)	объектілердің арнайы нүктелеріне байланыстыру үшін","d)	қабаттарды құру үшін."],
    correct: 2,
  },
  {
    title: 'Масштабтау мен панорамалау жұмыстары қандай менюдің қызметі болып табылады?',
    variants: ["a)	 Файл","b)	 Редакторлау","c)	 Инструменттер ","d)	 Вставка "],
    correct: 2,
  },
  {
    title: '6. Командалар топтамасының қайсысы Рисовать панеліне тиісті?',
    variants: ["a)	штрих, область, обрезка","b)	точка, эллипс, массив","c)	луч, линия, конструкции, блок","d)	полилиния, текст, объект"],
    correct: 2,
  },
  {
    title: '7. Оорs командасы нені білдіреді?',
    variants: ["a)	Команданы қайтып алу","b)	Өзгерту","c)	Өшіру","d)	Бұтарлау"],
    correct: 2,
  },
  {
    title: '8. Төмендегі кілттердің қайсысының объект таңдауға қатысы жоқ:',
    variants: ["a)	Wpolygon","b)	Relase","c)	Last","d)	Undo"],
    correct: 2,
  },
  {
    title: '9. Қай батырма Размер тақтасында жоқ?',
    variants: ["a) Linear measurement","b) Radius size","c) Diameter dimension","d) Angular dimension"],
    correct: 2,
  },
  {
    title: '10. Қай команда Изменить менюінің элементі емес? ',
    variants: ["a) Copy","b) Shift","c) Rotate","d) Length"],
    correct: 2,
  },
  {
    title: '11. Объектіге байлау режиміне көшу/алып тастау үшін қалып-күй жолының қай батырмасын басу керек?',
    variants: ["a) BINDING","b) GRID","c) ORTHO","d) POLAR"],
    correct: 2,
  },
  {
    title: '12. Қабаттың қандай қасиеті жоқ?',
    variants: ["a) Name","b) Visibility","c) Frozen","d) Layer area"],
    correct: 2,
  },
  {
    title: '13. Object binding құрал-саймандар панелінде қандай батырма жоқ?',
    variants: ["a)	Ең алыс нүктесіне байлау","b)	Нүктелерге байлау","c)	Объект центріне байлау","d)	Қисық сызықтың жанамасына байлау"],
    correct: 2,
  },
  {
    title: '14. Outline Setup => Object Binding сұхбат терезесінде объектіге байлаудың қандай түрі жоқ?',
    variants: ["a) End point","b) Midpoint","c) Center","d) Node"],
    correct: 2,
  },
  {
    title: '15. Outline Setup => Object Binding сұхбат терезесінде объектіге байлаудың қандай түрі жоқ?',
    variants: ["a) Near","b) Insert","c) Shift","d) Node"],
    correct: 2,
  },
  {
    title: '16. Ғ1 функциясының қызметі...',
    variants: ["a)	мәтіндік терезелерді ауыстыру","b)	графикалық терезелерді ауыстыру","c)	Autocad-тың анықтамалық жүйесін шақыру","d)	объектілерді байлау режимін іске қосу"],
    correct: 2,
  },
  {
    title: '17. Соңғы сызылған объектіні алып тастау командасы ',
    variants: ["a)	Undo","b)	Close","c)	Center","d)	Arc"],
    correct: 2,
  },
  {
    title: '18. Блокты жасақтау үшін...',
    variants: ["a) Draw=> Ellipse=> Make","b) Draw=> Block=> Make","c) Draw=> Arc=> Block","d) Draw=> Block=> Arc"],
    correct: 2,
  },
  {
    title: '19. Штрихтау командасы қай саймандар тақтасында орналасқан?',
    variants: ["a)	 Draw","b)	 Insert","c)	 Format","d)	 Edit"],
    correct: 2,
  },
  {
    title: '20. @ 80, 0 командасы нүктенің қандай координаталарын енгізеді?',
    variants: ["a)	Полярлық","b)	Салыстырмалы","c)	Абсолюттік ","d)	Абсолюттік және полярлық"],
    correct: 2,
  },
  {
    title: '21. Ағымдағы команданы тоқтату үшін қай батырма пайдаланылады?',
    variants: ["a)	Esc","b)	Delete","c)	Enter","d)	Ctrl+S"],
    correct: 2,
  },
  {
    title: '22. Pruning командасы қандай қызмет атқарады?',
    variants: ["a)	Объектіні екі нүктеде үзу","b)	Объектінің шетін қиып тастау","c)	Объект бөлігін берілген сызықтың бойымен қиып алу","d)	Объектінің шетін дөңгелектеу"],
    correct: 2,
  },
  {
    title: '23. OTRACK- дегеніміз не?',
    variants: ["a)	объектіні байлау","b)	торды бейнелеу не оны алып тастау","c)	ортогональды режимді қою не шығу","d)	объект байлауын іздеу"],
    correct: 2,
  }
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

function Test2() {
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

export default Test2