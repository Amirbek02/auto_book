import React, { useState } from 'react';


import './test.scss';
const questions = [
  {
    title: 'Графикалық редактор-бұл бағдарламалық жасақтама өнімі…',
    variants: ['a)	сызбаларды жасау кезінде ДК ресурстарын басқару',
    'b)	іс жүргізу процесінде мәтіндік ақпаратпен жұмыс',
    'c)	диаграммаларды құру','d)	Q. кескіндерді жасау және өңдеу'],
    correct: 2, // 0 a дұрыс, 1 b дұрыс, 2 с дұрыс, 3 d дұрыс
  },
  {
    title: 'AutoCAD файлдарында қандай кеңейтілім бар?',
    variants: ['a)	.doc',
    'b)	.dwg',
    'c)	.bmp',
    'd)	.jpeg'],
    correct: 0,
  },
  {
    title: '@ таңбасы …енгізу үшін қолданылады',
    variants: ['a)	нүктенің абсолютті декарттық координаттары',
    'b)	нүктенің абсолютті полярлық координаттары',
    'c)	нүктенің салыстырмалы декарттық координаттары',
    'd)	нүктенің салыстырмалы полярлық координаттары'],
    correct: 1,
  },
  {
    title: 'AutoCAD-та командасымен жұмыс істегенде объектілерді іріктеу реті қандай?',
    variants: [
      'a)	кесілген нысанды таңдаңыз',
      'b)	кесу жиектерін таңдаңыз.',
      'c)	кесу жиектерін таңдаңыз, содан кейін кесілген нысанды таңдаңыз. ',
      'd)	таңдау реті маңызды емес.'
    ],
    correct: 2,
  },
  {
    title: 'Блокты жасақтау үшін...',
    variants: ['a)	 Change => Property',
    'b)	 Draw => Area',
    'c)	 File => Block',
    'd)	 Draw => Block => Make'],
    correct: 2,
  },
  {
    title: 'Координаталарды объектіге байлау:',
    variants: ['a)	 Tools => Display order',
    'b)	 File => Object Binding',
    'c)	 Insert => Block',
    'd)	 Format => Layer'],
    correct: 2,
  },
  {
    title: 'Arс кілтінің қызметі:',
    variants: ['a)	Доға режиміне көшу',
    'b)	Поллилинияны кесіндісінен тұйықтау ',
    'c)	Кесіндінің жарты қалыңдығын енгізу ',
    'd)	Соңғы сызылған кесіндіні қайтып алу'],
    correct: 2,
  },
  {
    title: 'Стереть командасын пайдаланбай объектіні қай батырманың көмегімен өшіруге болады?',
    variants: ['a)	Insert ',
    'b)	Home',
    'c)	End',
    'd)	Delete'],
    correct: 2,
  },
  {
    title: 'Мәтіндік және графикалық терезелерді ауыстыру пернесі',
    variants: ['a)	F1 ',
    'b)	F2 ',
    'c)	F3 ',
    'd)	F4 '],
    correct: 2,
  },
  {
    title: 'Линия командасының кілттері',
    variants: ['a)	Hor, Ver, Ang, Bisect, Offset ',
    'b)	Angle, Center, Close ',
    'c)	Arc, Width, Undo',
    'd)	Line, Width'],
    correct: 2,
  },
  {
    title: 'Графикалық алаңшаның түсін қалай өзгертеміз?',
    variants: ['a)	Tools = > Options = > screen = > Color = > Color selection',
    'b)	Tools = > Color selection',
    'c)	Tools = > Color ',
    'd)	Draw => Line = > Color selection '],
    correct: 2,
  },
  {
    title: 'Берілген нүктеден өтетін горизанталь түзу сызу үшін LINE командасының қай кілті қажет?',
    variants: ['a)	Ver ',
    'b)	Hor',
    'c)	Ang  ',
    'd)	Bisect '],
    correct: 2,
  },
  {
    title: 'Менюдің SIZE элементі не үшін қолданылады?',
    variants: ['a)	сызба өлшемдерін қою ',
    'b)	блоктарды және объектілерді қою ',
    'c)	қабаттармен , түстермен сызықтар типтерімени жұмыс  ',
    'd)	сызу командалары '],
    correct: 2,
  },
  {
    title: 'Суреттен объекті өшіру командасы қай құрал-саймандар тақтасында?',
    variants: ['a)	Жиналмалы меню',
    'b)	Өзгерту ',
    'c)	Экрандық меню',
    'd)	Файл'],
    correct: 2,
  },
  {
    title: 'Соңғы команданы қайталау үшін қай түймені басу қажет?',
    variants: ['a) F1',
    'b) Enter',
    'c)	Esc',
    'd)	F2'],
    correct: 2,
  },
  {
    title: 'Линия командасының кілттері:',
    variants: ['a)	 Close, Undo',
    'b)	 Hor, Ver, Ang',
    'c)	 Angle, Center, Radius',
    'd)	 Length, Undo, Width'],
    correct: 2,
  },
  {
    title: 'Басы, соңы және бастапқы нүктесінен жүргізілген жанаманың көлбеулік бұрышы бойынша доғаны қай тәсілмен сызады?',
    variants: ['a)	 3Point',
    'b)	 Arccont',
    'c)	 St, End, Dir',
    'd)	 St, End, Rad'],
    correct: 2,
  },
  {
    title: 'Жиналмалы меню командасына жатпайтындар:',
    variants: ['a)	 Файл',
    'b)	 Формат',
    'c)	 Размер',
    'd)	 Вид'],
    correct: 2,
  },
  {
    title: 'Доға режимінде радиус енгізу кілтін көрсетіңіз.',
    variants: ['a)	 Undo',
    'b)	 Angle',
    'c)	 Line',
    'd)	 Radius'],
    correct: 2,
  },
  {
    title: 'Кесінді салуда нүктенің салыстырмалы полярлық координаталарын қалай енгізеді?',
    variants: ['a)	 х, у',
    'b)	 @ dx, dy',
    'c)	 x, y, z',
    'd)	 @ r, <A'],
    correct: 2,
  },
  {
    title: 'Дуга режимінде ChordLength  кілті қандай қызмет атқарады?',
    variants: ['a)	 хорданың ұзындығын енгізу',
    'b)	 Доғаның центрлік бұрышын енгізу',
    'c)	 Доғаның центрін енгізу ',
    'd)	 Доғаның жарты қалыңдығын енгізу'],
    correct: 2,
  },
  {
    title: 'Полигон командасы қай саймандар панелінде орналасқан?',
    variants: ['a)	 Рисовать',
    'b)	 Вставить',
    'c)	 Формат',
    'd)	 Изменить'],
    correct: 2,
  },

  // {
  //   title: '',
  //   variants: [],
  //   correct: 2,
  // },
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

function Test1() {
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

export default Test1