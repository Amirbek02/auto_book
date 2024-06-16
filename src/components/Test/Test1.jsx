import React, { useState } from 'react';

import './test.scss';
const questions = [
  {
    title:
      '1. Хабарламаларды есептеуіш машиналардың көмегімен сұрыптайтын жасанды тілдер тобы қалай аталады?',
    variants: ['А) программалық жасақтама', 'В) программалау', 'С) айнымалылар', 'Д) процедуралар'],
    correct: 1, // 0 a дұрыс, 1 b дұрыс, 2 с дұрыс, 3 d дұрыс
  },
  {
    title: '2. Құрылымдық программалау идеясы қашан пайда болды?',
    variants: ['А) 30 - жылдары', 'В) 40 - жылдары', 'С) 80 - жылдары', 'Д) 70 - жылдары'],
    correct: 3,
  },
  {
    title: '3. Өзара байланысқан негізгі нысандардан тұратын программалау тілі қалай аталады?',
    variants: [
      'А) модульдік программа',
      'В) нысанға бағытталған программалау тілі',
      'С) құрылымдық программалау',
      'Д) программалық жасақтама',
    ],
    correct: 1,
  },
  {
    title: '4. Компьютердегі программалық жасақтама құрамы қалай аталады?',
    variants: ['А) драйверлер', 'В) утилиттер', 'С) программалар', 'Д) программалық конфигурация'],
    correct: 3,
  },
  {
    title: '5. Программалау тілі дегеніміз не?',
    variants: [
      'A) Жоғары дәрежелі тілдердегі жазбаларды машиналық командалар тізбегіне айналдыратын жүйелік программа',
      'B) Программаны құрайтын жазбалар жүйесінде пайдаланылатын грамматикалық құрылыс синтаксисі мен семантикасын анықтайтын ережелер жиынтығы',
      'C) Есепті шешуде орындалатын әрекеттердің реттелген тізбегі',
    ],
    correct: 1,
  },
  {
    title: '6. Программалау жүйелерінде тілді іске асыру құралдарын атап көрсет',
    variants: [
      'A) машиналық - бағытталған, машиналық - тәуелсіз',
      'B) компиляторлар, интерпретаторлар',
      'C) жүзеге асатын, жүзеге аспайтын',
    ],
    correct: 1,
  },
  {
    title: '7. Процедуралық программалауды қай жылы кім ұсынған?',
    variants: ['A) 40ж. Фон Нейман', 'B) 40ж. Паскаль', 'C) 50ж. Фортон'],
    correct: 2,
  },
  {
    title: '8. Обьектті - бағытталған программалаудың типін негізін салушы?',
    variants: ['A) Алан Кэй ', 'B) Бейсик', 'C) Липс'],
    correct: 0,
  },
  {
    title: '9. Қасиет дегеніміз -...',
    variants: ['A) өлшемі', 'B) мәндер өзгерту', 'C) өлшемі, жағдайы, түсі, жазбасы'],
    correct: 2,
  },
  {
    title: '10. Тізім басқару элементі?',
    variants: ['A) Text', 'B) ListBox', 'C) Combobox'],
    correct: 1,
  },
  {
    title: '11. Қандай қосымша бетте жаңа жоба үшін шаблон таңдалады?',
    variants: ['A) New', 'B) Existing', 'C) Recent'],
    correct: 0,
  },
  {
    title: '12. Visual Basic программасында Project (жоба) атын өзгертуге бола ма?',
    variants: ['A) Болмайды', 'B) болады', 'C) дұрыс жауабы жоқ'],
    correct: 1,
  },
  {
    title: '13. New Project терезесі неше қосымша беттен тұрады?',
    variants: ['A)	1', 'B)	2', 'C) 3'],
    correct: 0,
  },
  {
    title: '14. Visual Basic терезесінің жоғарғы жағында орналасқан мәтін жолын қалай атайды?',
    variants: ['A) Project менюі', 'B) Басты меню', 'C) File менюі', 'D) Edit менюі'],
    correct: 1,
  },
  {
    title: '15. Пішінмен байланысты бағдарламалық кодты қарау үшін қандай батырманы басу керек?',
    variants: ['A) View', 'B) Toolbars', 'C) View object', 'D) View Code'],
    correct: 3,
  },
  {
    title: '16. Visual Basic - тің негізгі объектісі?',
    variants: ['A)	Жол', 'B) Қасиет', 'C) Пішін', 'D) Файл'],
    correct: 1,
  },
  {
    title: '17. New Project терезесі қандай қосымша беттерден тұрады?',
    variants: [
      'A) Existing, Project Wizart',
      'B) Recent, Visual Basic, New',
      'C) New, Existing, Recent',
    ],
    correct: 2,
  },
  {
    title:
      '18. Басты менюдің тура астында Visual Basic инструменттер панелі болмаса, оны қалай шығаруға болады?',
    variants: [
      'A) View - Toolbars',
      'B) View - Toolbars - Standart ',
      'C) File - Toolbars - Standart',
    ],
    correct: 1,
  },
  {
    title: '19. Visual Basic программасын іске қосу жолы?',
    variants: [
      'A) Пуск – Все программмы - Visual Basic 6. 0 ',
      'B) Мои документы - Visual Basic',
      'C) Пуск - Стандартные - Visual Basic',
    ],
    correct: 0,
  },
  {
    title: '20. Жобаға жаңа форма қосу командасы.',
    variants: ['A) Add Form', 'B) File – Add Form', 'C) Project – Add Form'],
    correct: 2,
  },
  {
    title: '21. Visual Basic программалау тілінде қандай меню жобаның жүрегі болып саналады?',
    variants: ['A) Project менюі ', 'B) Басты меню', 'C) File менюі', 'D) Edit менюі'],
    correct: 0,
  },
  {
    title: '22. Қолданбаны құрайтын барлық объектілер қайда бірігеді?',
    variants: [
      'A) Инструменттер панеліне',
      'B) Басты менюге',
      'C) Жоба терезесіне',
      'D) Қасиеттер терезесіне',
    ],
    correct: 2,
  },
  {
    title: '23. Мәтін енгізу үшін қолданылатын элемент',
    variants: ['А) TextBox', 'B) Label', 'C) Image', 'D) Times'],
    correct: 0,
  },
  {
    title: '24. Project – Add Form командасы арқылы жобаға не қосылады?',
    variants: ['A) Жаңа элемент', 'B) Жаңа пішін', 'C) Жаңа жоба', 'D) Жаңа мәтін'],
    correct: 1,
  },
  {
    title: '25. Программалық тілді іске асыру құралдары нешеге бөлінеді?',
    variants: ['A)	1', 'B)	2', 'C)	3', 'D)	4'],
    correct: 1,
  },

  // {
  //   title: '',
  //   variants: [],
  //   correct: 2,
  // },
];

function Result({ correct }) {
  return (
    <div className="result">
      <img src="https://cdn-icons-png.flaticon.com/512/2278/2278992.png" />
      <h2>
        Сіз {questions.length} сұрақтын {correct} сұрағына дұрыс жауап бердіңіз{' '}
      </h2>
      <a href="/test">
        {' '}
        <button>Тестті аяқтау</button>
      </a>
    </div>
  );
}

function Game({ question, onClickVariants, step }) {
  return (
    <>
      <div className="progress">
        <div
          style={{ width: `${Math.round((step / questions.length) * 100)}%` }}
          className="progress__inner"></div>
      </div>
      <h1>{question.title}</h1>
      <ul>
        {question.variants.map((text, index) => (
          <li onClick={() => onClickVariants(index)} key={text}>
            {text}
          </li>
        ))}
      </ul>
    </>
  );
}

function Test1() {
  const [step, setStep] = useState(0);
  const [correct, setCorrect] = useState(0);
  const question = questions[step];
  const onClickVariants = (index) => {
    setStep(step + 1);

    if (index === question.correct) {
      setCorrect(correct + 1);
    }
  };
  return (
    <div className="test">
      {step !== questions.length ? (
        <Game step={step} question={question} onClickVariants={onClickVariants} />
      ) : (
        <Result correct={correct} />
      )}
    </div>
  );
}

export default Test1;
