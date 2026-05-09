"""
Regenerate raw QA response files with project-aligned distribution and
more topic-aware interview answers.

This script intentionally does not call an external LLM. It is a deterministic
quality pass for the synthetic QA seed layer:

- backend: 150 per score band = 600
- frontend: 100 per score band = 400
- data_ai: 150 per score band = 600
- devops: 100 per score band = 400

Total: 2,000 QA pairs, matching configs/data_gen.yaml and CLAUDE.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = Path("data/raw/qa_prompts")
RESPONSES_DIR = Path("data/raw/qa_responses")

TARGETS_PER_BAND = {
    "backend": 150,
    "frontend": 100,
    "data_ai": 150,
    "devops": 100,
}

MIN_LENGTH = {
    "1-3": 100,
    "4-6": 150,
    "7-8": 200,
    "9-10": 250,
}


@dataclass(frozen=True)
class SeedQuestion:
    question_type: str
    question: str


@dataclass(frozen=True)
class TopicProfile:
    name: str
    correct: str
    shallow: str
    wrong: str
    tradeoff: str


PROFILES: list[tuple[tuple[str, ...], TopicProfile]] = [
    (
        ("TCP", "handshake"),
        TopicProfile(
            "TCP 3-way handshake",
            "클라이언트가 SYN을 보내고 서버가 SYN-ACK로 응답한 뒤 클라이언트가 ACK를 보내 연결을 확정하는 과정입니다. 양쪽의 수신 가능 여부와 초기 시퀀스 번호를 맞추기 위해 필요합니다.",
            "연결 전에 서로 통신 가능한지 확인하는 절차라는 정도는 알고 있습니다. SYN, ACK 같은 패킷이 오간다고 알고 있지만 각 단계의 의미나 시퀀스 번호까지는 자세히 설명하기 어렵습니다.",
            "TCP는 빠른 전송을 위해 연결 없이 데이터를 보내는 방식이고, handshake는 중간에 패킷을 압축하는 과정이라고 알고 있습니다. UDP와 큰 차이는 없다고 생각합니다.",
            "연결 수립 비용이 생기므로 짧은 요청이 많은 환경에서는 커넥션 재사용, keep-alive, TLS 비용까지 함께 봐야 합니다.",
        ),
    ),
    (
        ("인덱스",),
        TopicProfile(
            "데이터베이스 인덱스",
            "테이블 전체를 훑지 않고 조건에 맞는 행을 빠르게 찾기 위한 별도 자료구조입니다. 보통 B-Tree 계열을 많이 쓰며 WHERE, JOIN, ORDER BY에 자주 쓰는 컬럼에 효과적입니다.",
            "조회 속도를 빠르게 해주는 기능이라는 점은 알고 있습니다. 다만 어떤 컬럼에 걸어야 하는지, 실행 계획을 어떻게 확인하는지는 구체적으로 설명하기 어렵습니다.",
            "인덱스는 데이터를 메모리에 전부 복사해두는 기능이라 많이 만들수록 항상 빨라진다고 알고 있습니다. 저장 공간이나 쓰기 성능 문제는 거의 없다고 생각합니다.",
            "쓰기 작업마다 인덱스도 갱신되므로 INSERT, UPDATE 비용과 저장 공간이 늘어납니다. 카디널리티, 복합 인덱스 컬럼 순서, 실제 쿼리 패턴을 함께 봐야 합니다.",
        ),
    ),
    (
        ("트랜잭션",),
        TopicProfile(
            "트랜잭션",
            "여러 데이터 변경을 하나의 작업 단위로 묶어 모두 성공하거나 모두 실패하게 하는 개념입니다. Spring에서는 보통 @Transactional 프록시가 커밋과 롤백을 관리합니다.",
            "여러 DB 작업을 묶어서 처리한다는 정도로 이해하고 있습니다. 실패하면 롤백된다는 것은 알지만 전파 옵션이나 격리 수준은 정확히 설명하기 어렵습니다.",
            "트랜잭션은 SQL을 더 빨리 실행하는 옵션이라고 알고 있습니다. 실패해도 DB가 알아서 적당히 저장하므로 롤백 같은 건 크게 신경 쓰지 않아도 된다고 생각합니다.",
            "자가 호출 시 프록시를 거치지 않는 문제, checked exception 롤백 정책, 격리 수준에 따른 동시성 현상을 함께 고려해야 합니다.",
        ),
    ),
    (
        ("REST", "GraphQL"),
        TopicProfile(
            "REST API와 GraphQL",
            "REST는 리소스와 HTTP 메서드 중심으로 여러 엔드포인트를 설계하고, GraphQL은 클라이언트가 필요한 필드를 쿼리로 지정합니다.",
            "REST는 URL을 나누고 GraphQL은 한 번에 데이터를 가져온다는 정도로 알고 있습니다. 장단점은 대략 알지만 캐싱이나 쿼리 복잡도까지는 잘 모릅니다.",
            "REST와 GraphQL은 이름만 다르고 둘 다 JSON을 보내는 방식이라 차이가 거의 없다고 생각합니다. GraphQL은 항상 REST보다 빠른 방식으로 알고 있습니다.",
            "REST는 HTTP 캐싱과 단순성이 장점이고, GraphQL은 과다 조회를 줄일 수 있지만 N+1, 권한 처리, 쿼리 비용 제한을 설계해야 합니다.",
        ),
    ),
    (
        ("프로세스", "스레드"),
        TopicProfile(
            "프로세스와 스레드",
            "프로세스는 독립된 메모리 공간을 가진 실행 단위이고 스레드는 같은 프로세스 안에서 자원을 공유하는 실행 흐름입니다.",
            "프로세스가 더 큰 단위이고 스레드는 그 안에서 실행된다는 정도는 알고 있습니다. 메모리 공유나 동기화 문제는 깊게 설명하기 어렵습니다.",
            "프로세스와 스레드는 거의 같은 말이고, 스레드는 CPU 속도를 올리는 설정이라고 알고 있습니다. 많이 만들수록 서버가 무조건 빨라진다고 생각합니다.",
            "스레드는 생성 비용이 작고 공유가 쉬운 대신 race condition, deadlock, context switching 비용이 있어 풀 크기와 동기화 전략이 중요합니다.",
        ),
    ),
    (
        ("Redis", "캐시"),
        TopicProfile(
            "Redis 캐시",
            "Redis는 메모리 기반 저장소로 자주 조회되는 데이터를 빠르게 제공하거나 세션, 분산 락, 메시지 처리 등에 활용할 수 있습니다.",
            "DB 조회를 줄이기 위해 Redis를 캐시로 쓴다는 정도는 알고 있습니다. TTL이나 캐시 무효화 전략은 구체적으로 설명하기 어렵습니다.",
            "Redis는 데이터베이스를 완전히 대체하는 저장소이고 메모리라서 데이터가 절대 사라지지 않는다고 알고 있습니다. 캐시는 오래 둘수록 좋다고 생각합니다.",
            "캐시 히트율, TTL, stampede, stale data를 함께 봐야 하며 원본 DB와의 일관성 요구에 따라 write-through, cache-aside 같은 방식을 선택해야 합니다.",
        ),
    ),
    (
        ("JPA",),
        TopicProfile(
            "JPA",
            "JPA는 객체와 관계형 DB 테이블 사이의 매핑을 도와 반복적인 SQL 작성을 줄이는 ORM 기술입니다. 영속성 컨텍스트와 지연 로딩을 이해해야 합니다.",
            "SQL을 직접 덜 작성하게 해주는 기술이라고 알고 있습니다. Entity나 Repository는 써봤지만 영속성 컨텍스트나 N+1 문제는 깊게 설명하기 어렵습니다.",
            "JPA는 SQL보다 무조건 빠른 데이터베이스라고 알고 있습니다. 쿼리를 몰라도 자동으로 최적화해주기 때문에 실행 계획을 볼 필요는 없다고 생각합니다.",
            "편의성 뒤에 생성 SQL이 숨어 있으므로 fetch join, batch size, 변경 감지, 트랜잭션 범위를 함께 확인해야 운영 성능 문제를 줄일 수 있습니다.",
        ),
    ),
    (
        ("React", "state", "props"),
        TopicProfile(
            "React state와 props",
            "props는 부모가 자식에게 전달하는 읽기 중심 데이터이고 state는 컴포넌트 내부에서 변경되며 렌더링을 유발하는 상태입니다.",
            "props는 전달받는 값이고 state는 바뀌는 값이라는 정도로 알고 있습니다. 상태를 어디에 둘지나 렌더링 영향까지는 명확히 설명하기 어렵습니다.",
            "state와 props는 둘 다 변수라서 아무 곳에서나 수정해도 된다고 생각합니다. props도 자식 컴포넌트에서 직접 바꾸면 화면이 갱신되는 걸로 알고 있습니다.",
            "상태 위치를 잘못 잡으면 prop drilling, 불필요한 렌더링, 동기화 문제가 생기므로 소유권과 변경 주체를 기준으로 설계해야 합니다.",
        ),
    ),
    (
        ("브라우저 렌더링",),
        TopicProfile(
            "브라우저 렌더링",
            "브라우저가 HTML을 파싱해 DOM을 만들고 CSSOM과 결합해 렌더 트리를 만든 뒤 layout, paint, composite 과정을 거쳐 화면에 표시하는 흐름입니다.",
            "HTML과 CSS를 읽어서 화면을 그리는 과정이라는 정도는 알고 있습니다. DOM, CSSOM, layout, paint 단계가 어떻게 이어지는지는 자세히 설명하기 어렵습니다.",
            "브라우저 렌더링은 서버가 HTML을 이미지로 만들어 보내는 과정이라고 알고 있습니다. JavaScript 실행이나 CSS 계산은 크게 관련 없다고 생각합니다.",
            "DOM 변경과 스타일 계산이 잦으면 reflow와 repaint 비용이 커지므로 레이아웃 변경 범위, 애니메이션 속성, 렌더링 타이밍을 함께 봐야 합니다.",
        ),
    ),
    (
        ("이벤트 루프",),
        TopicProfile(
            "JavaScript 이벤트 루프",
            "콜 스택이 비었을 때 태스크 큐와 마이크로태스크 큐의 작업을 꺼내 실행해 싱글 스레드에서도 비동기 처리를 가능하게 하는 구조입니다.",
            "비동기 처리를 도와주는 구조라는 정도는 알고 있습니다. Promise와 setTimeout의 실행 순서 차이나 마이크로태스크 큐는 헷갈립니다.",
            "이벤트 루프는 여러 CPU 스레드를 만들어 JavaScript를 병렬 실행하는 기능이라고 알고 있습니다. 그래서 setTimeout은 항상 정확한 시간에 실행된다고 생각합니다.",
            "마이크로태스크가 과도하면 렌더링이 밀릴 수 있고, 긴 동기 작업은 UI를 막으므로 작업 분할과 Web Worker 사용 여부를 함께 판단해야 합니다.",
        ),
    ),
    (
        ("React Hook",),
        TopicProfile(
            "React Hook",
            "Hook은 함수 컴포넌트에서 상태와 생명주기 관련 기능을 사용할 수 있게 해줍니다. useEffect는 렌더링 이후 부수 효과를 처리할 때 사용합니다.",
            "함수 컴포넌트에서 state나 effect를 쓰게 해주는 기능이라는 정도는 알고 있습니다. dependency 배열이나 cleanup 처리는 아직 헷갈립니다.",
            "Hook은 React 성능을 자동으로 올려주는 플러그인이고 아무 조건 없이 if문 안에서 호출해도 된다고 알고 있습니다. useEffect는 렌더링 전에 항상 실행된다고 생각합니다.",
            "의존성 배열 누락은 stale closure나 무한 반복을 만들 수 있고, 구독이나 타이머는 cleanup으로 해제해야 메모리 누수를 줄일 수 있습니다.",
        ),
    ),
    (
        ("접근성",),
        TopicProfile(
            "웹 접근성",
            "접근성은 장애 여부나 사용 환경과 관계없이 사용자가 서비스를 이용할 수 있게 하는 품질 기준입니다. 시맨틱 HTML, 키보드 탐색, 대체 텍스트가 중요합니다.",
            "여러 사용자가 화면을 잘 볼 수 있게 하는 것이라는 정도는 알고 있습니다. ARIA나 키보드 포커스 관리 같은 세부 구현은 아직 익숙하지 않습니다.",
            "접근성은 디자인을 예쁘게 만드는 작업이라고 알고 있습니다. 실제 사용자가 적으면 우선순위가 낮고, 나중에 CSS만 바꾸면 해결된다고 생각합니다.",
            "법적 요구와 사용자 경험뿐 아니라 검색, 테스트, 유지보수에도 영향을 주므로 초기 컴포넌트 설계부터 role, label, focus 흐름을 확인해야 합니다.",
        ),
    ),
    (
        ("TypeScript",),
        TopicProfile(
            "TypeScript",
            "TypeScript는 JavaScript에 정적 타입을 더해 컴파일 단계에서 오류를 줄이고 IDE 자동완성과 리팩토링 안정성을 높이는 언어입니다.",
            "타입을 붙여서 오류를 줄이는 도구라는 정도는 알고 있습니다. 제네릭이나 유니언 타입을 설계에 어떻게 활용하는지는 아직 부족합니다.",
            "TypeScript는 브라우저에서 JavaScript보다 빠르게 실행되는 언어라고 알고 있습니다. 타입을 any로 두면 더 유연해서 좋은 코드라고 생각합니다.",
            "타입이 런타임 검증을 대신하지는 않으므로 API 응답 검증, strict 옵션, 타입 추론과 명시 타입의 균형을 함께 고려해야 합니다.",
        ),
    ),
    (
        ("클로저",),
        TopicProfile(
            "JavaScript 클로저",
            "클로저는 함수가 선언될 당시의 외부 스코프 변수를 기억해 이후에도 접근할 수 있는 특성입니다. 캡슐화나 콜백에서 자주 활용됩니다.",
            "함수 안에서 바깥 변수를 사용할 수 있는 개념이라는 정도는 알고 있습니다. 실행 컨텍스트나 메모리 관점은 자세히 설명하기 어렵습니다.",
            "클로저는 함수를 강제로 종료하는 문법이라고 알고 있습니다. 변수를 복사해서 쓰기 때문에 메모리와는 별로 관련 없다고 생각합니다.",
            "의도치 않게 오래 참조되는 변수는 메모리 누수나 stale state를 만들 수 있어 React callback, timer, event listener에서 특히 주의해야 합니다.",
        ),
    ),
    (
        ("Code Splitting",),
        TopicProfile(
            "Code Splitting",
            "Code Splitting은 번들을 여러 조각으로 나눠 필요한 시점에 로드해 초기 로딩 비용을 줄이는 최적화 기법입니다.",
            "코드를 나눠서 로딩을 빠르게 하는 방식이라는 정도는 알고 있습니다. 라우트 단위 분리나 lazy loading 기준은 구체적으로 설명하기 어렵습니다.",
            "Code Splitting은 파일을 여러 개로 저장하면 자동으로 성능이 좋아지는 방식이라고 알고 있습니다. 네트워크 요청이 늘어나는 문제는 없다고 생각합니다.",
            "초기 번들 크기, 캐시 전략, chunk 수, fallback UI, prefetch 여부를 함께 봐야 실제 사용자 지표 개선으로 이어집니다.",
        ),
    ),
    (
        ("Reflow", "Repaint"),
        TopicProfile(
            "Reflow와 Repaint",
            "Reflow는 요소의 위치와 크기를 다시 계산하는 과정이고 Repaint는 계산된 영역을 다시 그리는 과정입니다. Reflow가 보통 더 비쌉니다.",
            "둘 다 화면을 다시 그릴 때 생기는 비용이라는 정도는 알고 있습니다. 어떤 CSS 속성이 어느 단계에 영향을 주는지는 잘 모릅니다.",
            "Reflow와 Repaint는 React에서만 발생하는 상태 관리 기능이라고 알고 있습니다. 화면이 바뀌면 항상 같은 비용이 든다고 생각합니다.",
            "layout을 바꾸는 속성보다 transform, opacity를 활용하면 비용을 줄일 수 있고, DOM 읽기와 쓰기를 섞으면 layout thrashing이 생길 수 있습니다.",
        ),
    ),
    (
        ("Hoisting",),
        TopicProfile(
            "JavaScript Hoisting",
            "Hoisting은 변수와 함수 선언이 실행 컨텍스트 생성 단계에서 먼저 등록되는 것처럼 동작하는 특성입니다. var, let, const, 함수 선언식의 차이가 있습니다.",
            "선언이 위로 올라가는 것처럼 동작한다는 정도는 알고 있습니다. TDZ나 함수 표현식과 선언식 차이는 아직 헷갈립니다.",
            "Hoisting은 브라우저가 코드를 빠르게 실행하려고 변수 값을 자동으로 위로 복사하는 기능이라고 알고 있습니다. let과 var는 차이가 거의 없다고 생각합니다.",
            "let과 const는 TDZ 때문에 선언 전 접근이 오류가 나며, 이 차이를 모르면 초기화 순서 버그나 예측하기 어려운 코드가 생길 수 있습니다.",
        ),
    ),
    (
        ("XSS",),
        TopicProfile(
            "XSS",
            "XSS는 공격자가 삽입한 스크립트가 사용자의 브라우저에서 실행되는 공격입니다. 입력 검증, 출력 인코딩, CSP 등으로 방어합니다.",
            "사용자 입력으로 스크립트가 실행되는 보안 문제라는 정도는 알고 있습니다. 저장형과 반사형 차이나 CSP 설정은 자세히 모릅니다.",
            "XSS는 서버 CPU를 많이 쓰게 하는 공격이라고 알고 있습니다. HTTPS를 쓰면 스크립트 공격은 자동으로 막힌다고 생각합니다.",
            "React의 기본 escaping을 우회하는 dangerouslySetInnerHTML, 외부 HTML 렌더링, 토큰 저장 위치까지 함께 점검해야 합니다.",
        ),
    ),
    (
        ("CSRF",),
        TopicProfile(
            "CSRF",
            "CSRF는 사용자가 인증된 상태를 악용해 의도하지 않은 요청을 보내게 만드는 공격입니다. CSRF 토큰과 SameSite 쿠키로 방어합니다.",
            "로그인된 사용자의 권한을 악용하는 공격이라는 정도는 알고 있습니다. SameSite나 토큰 검증 흐름은 구체적으로 설명하기 어렵습니다.",
            "CSRF는 비밀번호를 무작위로 대입하는 공격이라고 알고 있습니다. 프론트에서 alert만 띄우면 대부분 막을 수 있다고 생각합니다.",
            "쿠키 기반 인증에서는 특히 중요하며 CORS와 목적이 다르므로 Origin 검증, SameSite 설정, 상태 변경 메서드 보호를 함께 적용해야 합니다.",
        ),
    ),
    (
        ("CORS",),
        TopicProfile(
            "CORS",
            "브라우저의 동일 출처 정책 때문에 다른 출처의 리소스 접근을 서버가 허용했는지 확인하는 메커니즘입니다. 응답 헤더와 preflight가 핵심입니다.",
            "다른 도메인 API를 호출할 때 생기는 보안 설정이라는 정도는 알고 있습니다. 어떤 헤더가 필요한지나 credentials 옵션은 자세히 모릅니다.",
            "CORS는 서버 장애나 네트워크 속도 문제라고 알고 있습니다. 프론트에서 mode를 no-cors로 바꾸면 대부분 해결되기 때문에 서버 설정은 중요하지 않다고 생각합니다.",
            "Origin, Method, Header, credentials 조합을 정확히 맞춰야 하며 와일드카드 허용은 보안 위험이 있어 운영 환경에서는 허용 범위를 좁혀야 합니다.",
        ),
    ),
    (
        ("Virtual DOM",),
        TopicProfile(
            "Virtual DOM",
            "UI 변경 내용을 메모리상의 가상 트리에 먼저 반영하고 이전 결과와 비교해 실제 DOM 변경을 줄이기 위한 React의 렌더링 추상화입니다.",
            "실제 DOM보다 빠른 중간 객체라는 정도로 알고 있습니다. diffing이나 reconciliation이 어떻게 동작하는지는 자세히 설명하기 어렵습니다.",
            "Virtual DOM은 브라우저 DOM을 없애고 HTML을 서버에서만 그리는 기술이라고 알고 있습니다. 그래서 쓰면 성능이 항상 좋아진다고 생각합니다.",
            "항상 빠른 만능 기술은 아니며 key 설계, 컴포넌트 분리, memoization, 상태 변경 범위가 맞지 않으면 불필요한 렌더링이 여전히 발생합니다.",
        ),
    ),
    (
        ("SSR", "CSR"),
        TopicProfile(
            "SSR과 CSR",
            "SSR은 서버에서 HTML을 만들어 보내 초기 표시와 SEO에 유리하고, CSR은 브라우저에서 JavaScript로 화면을 구성해 이후 상호작용이 유연합니다.",
            "SSR은 서버, CSR은 클라이언트에서 화면을 그린다는 정도는 알고 있습니다. SEO나 hydration 같은 세부 차이는 깊게 설명하기 어렵습니다.",
            "SSR은 CSS를 서버에 저장하는 방식이고 CSR은 캐시를 쓰는 방식이라고 알고 있습니다. 둘은 성능 차이가 거의 없고 취향 문제라고 생각합니다.",
            "SSR은 서버 부하와 hydration 비용이 있고 CSR은 초기 로딩과 SEO가 약할 수 있어 페이지 성격, 캐싱, TTFB, LCP를 기준으로 선택해야 합니다.",
        ),
    ),
    (
        ("WebSocket",),
        TopicProfile(
            "WebSocket",
            "HTTP로 handshake를 한 뒤 하나의 연결을 유지하면서 서버와 클라이언트가 양방향으로 메시지를 주고받는 통신 방식입니다.",
            "실시간 통신에 쓰는 기술이라는 정도는 알고 있습니다. HTTP polling과의 차이나 연결 관리 방식은 정확히 설명하기 어렵습니다.",
            "WebSocket은 HTTP 요청을 더 빠르게 압축하는 기능이라고 알고 있습니다. 연결을 유지하지 않고 요청할 때마다 새로 열리는 방식이라고 생각합니다.",
            "연결 수가 많아지면 서버 자원, heartbeat, 재연결, 메시지 순서, 인증 만료 처리를 설계해야 하며 단순 알림은 SSE도 대안이 될 수 있습니다.",
        ),
    ),
    (
        ("LocalStorage", "SessionStorage"),
        TopicProfile(
            "LocalStorage와 SessionStorage",
            "둘 다 브라우저에 key-value 데이터를 저장하는 Web Storage입니다. LocalStorage는 명시적으로 지우기 전까지 남고 SessionStorage는 탭 세션이 끝나면 사라집니다.",
            "브라우저에 데이터를 저장하는 기능이라는 정도는 알고 있습니다. 만료 시점이나 탭 단위 차이, 보안상 주의점은 자세히 설명하기 어렵습니다.",
            "LocalStorage는 서버 DB에 저장되고 SessionStorage는 쿠키를 암호화하는 기능이라고 알고 있습니다. 둘 다 민감 정보를 저장해도 안전하다고 생각합니다.",
            "둘 다 JavaScript로 접근 가능하므로 XSS에 취약할 수 있어 access token 같은 민감 정보 저장은 신중해야 합니다.",
        ),
    ),
    (
        ("오버피팅",),
        TopicProfile(
            "오버피팅",
            "모델이 학습 데이터의 일반 패턴보다 노이즈까지 외워 검증 또는 실제 데이터 성능이 떨어지는 현상입니다.",
            "학습 데이터에 너무 맞춰지는 현상이라는 정도는 알고 있습니다. 정규화나 드롭아웃 같은 해결 방법은 들어봤지만 언제 쓰는지는 잘 모릅니다.",
            "오버피팅은 정확도가 100%가 되어 모델이 완성된 상태라고 알고 있습니다. 데이터가 많으면 무조건 생기고 에폭을 늘리면 해결된다고 생각합니다.",
            "train/validation 곡선을 함께 보고 조기 종료, 정규화, 데이터 증강, 모델 복잡도 조절을 선택해야 하며 데이터 누수 여부도 확인해야 합니다.",
        ),
    ),
    (
        ("파인튜닝", "프롬프트"),
        TopicProfile(
            "파인튜닝과 프롬프트 엔지니어링",
            "파인튜닝은 모델 가중치를 추가 학습해 특정 태스크나 출력 형식에 맞추는 방식이고, 프롬프트 엔지니어링은 입력 지시문을 설계해 원하는 출력을 유도하는 방식입니다.",
            "둘 다 모델 답변을 좋게 만드는 방법이라는 정도는 알고 있습니다. 다만 비용, 데이터 필요량, 적용 기준 차이는 명확히 설명하기 어렵습니다.",
            "파인튜닝은 프롬프트를 길게 쓰는 방식이고 프롬프트 엔지니어링은 모델을 다시 학습시키는 방식이라고 알고 있습니다. 둘은 거의 같은 작업이라고 생각합니다.",
            "프롬프트로 해결 가능한 형식 문제인지, 지속적으로 같은 행동을 학습해야 하는지, 데이터 품질과 운영 비용을 기준으로 선택해야 합니다.",
        ),
    ),
    (
        ("정확도", "Accuracy"),
        TopicProfile(
            "Accuracy",
            "Accuracy는 전체 샘플 중 맞게 예측한 비율입니다. 클래스 불균형이 크면 높은 정확도라도 중요한 소수 클래스를 놓칠 수 있습니다.",
            "전체 중 얼마나 맞혔는지를 보는 지표라는 정도는 알고 있습니다. 불균형 데이터에서 왜 문제가 되는지는 예시로 설명하기 어렵습니다.",
            "Accuracy가 90%면 대부분 실무에서 바로 쓸 수 있다고 알고 있습니다. 데이터 분포나 오탐 비용은 크게 중요하지 않다고 생각합니다.",
            "문제 비용 구조에 따라 precision, recall, F1, AUC를 함께 보고 임계값을 조정해야 실제 의사결정에 맞는 평가가 됩니다.",
        ),
    ),
    (
        ("분류", "회귀"),
        TopicProfile(
            "분류와 회귀",
            "분류는 정해진 클래스나 범주를 예측하고, 회귀는 연속적인 수치 값을 예측하는 문제입니다.",
            "분류는 카테고리, 회귀는 숫자를 예측한다는 정도는 알고 있습니다. 손실 함수나 평가 지표 차이는 자세히 설명하기 어렵습니다.",
            "분류는 숫자를 예측하고 회귀는 카테고리를 맞추는 문제라고 알고 있습니다. 둘 다 모델이 예측하는 거라 구현 차이는 거의 없다고 생각합니다.",
            "분류는 cross entropy, 회귀는 MSE 같은 손실을 주로 쓰며 출력층, 평가 지표, 오류 비용이 문제 정의에 따라 달라집니다.",
        ),
    ),
    (
        ("Train", "Validation", "Test"),
        TopicProfile(
            "Train Validation Test split",
            "Train은 모델 학습, validation은 튜닝과 선택, test는 최종 일반화 성능 확인에 사용합니다.",
            "데이터를 학습용과 평가용으로 나눈다는 정도는 알고 있습니다. validation과 test를 왜 분리해야 하는지는 명확히 설명하기 어렵습니다.",
            "데이터를 많이 학습시키는 게 좋으니 test 데이터도 학습에 넣어도 된다고 생각합니다. 나누는 건 파일 관리 편의를 위한 것이라고 알고 있습니다.",
            "test를 반복적으로 참고하면 평가가 오염되므로 시간 순서, 사용자 단위, 데이터 누수 가능성을 고려해 분리 기준을 설계해야 합니다.",
        ),
    ),
    (
        ("Gradient Descent",),
        TopicProfile(
            "Gradient Descent",
            "손실 함수를 줄이는 방향으로 파라미터를 조금씩 업데이트하는 최적화 방법입니다. 기울기와 learning rate가 업데이트 크기를 결정합니다.",
            "오차를 줄이기 위해 가중치를 조정하는 방법이라는 정도는 알고 있습니다. learning rate나 local minimum 문제는 자세히 설명하기 어렵습니다.",
            "Gradient Descent는 데이터를 정렬해서 빠르게 찾는 알고리즘이라고 알고 있습니다. 학습률은 클수록 항상 빨리 좋아진다고 생각합니다.",
            "학습률이 너무 크면 발산하고 너무 작으면 느리게 수렴하므로 optimizer, scheduler, gradient clipping을 함께 고려해야 합니다.",
        ),
    ),
    (
        ("데이터 누수", "Data Leakage"),
        TopicProfile(
            "Data Leakage",
            "데이터 누수는 학습 시점에 알 수 없어야 하는 정보가 학습이나 검증에 들어가 평가 성능이 비정상적으로 높아지는 문제입니다.",
            "평가 데이터가 학습에 섞이는 문제라는 정도는 알고 있습니다. 전처리나 시간 기준에서 생기는 누수는 구체적으로 설명하기 어렵습니다.",
            "데이터 누수는 파일이 외부로 유출되는 보안 사고라고 알고 있습니다. 모델 성능 평가와는 별로 관련 없다고 생각합니다.",
            "스케일링, 인코딩, feature 생성도 train 기준으로만 fit해야 하며 시간 순서가 있는 데이터는 미래 정보가 섞이지 않도록 검증해야 합니다.",
        ),
    ),
    (
        ("RAG",),
        TopicProfile(
            "RAG",
            "검색으로 관련 문서를 가져온 뒤 생성 모델이 그 근거를 바탕으로 답변하게 하는 방식입니다. 최신 지식이나 사내 문서 활용에 유리합니다.",
            "모델이 외부 문서를 참고해 답변한다는 정도는 알고 있습니다. 검색 품질이나 chunking, reranking까지는 자세히 설명하기 어렵습니다.",
            "RAG는 모델이 인터넷을 직접 검색해서 항상 최신 정답을 보장하는 기능이라고 알고 있습니다. 붙이기만 하면 hallucination은 없어지는 걸로 생각합니다.",
            "검색 실패가 곧 생성 실패로 이어질 수 있어 문서 분할, 임베딩 모델, 재랭킹, 출처 표시, 권한 필터링을 함께 설계해야 합니다.",
        ),
    ),
    (
        ("Attention",),
        TopicProfile(
            "Transformer Attention",
            "입력 토큰들이 서로 얼마나 관련 있는지 가중치를 계산해 중요한 정보를 더 반영하는 메커니즘입니다. Query, Key, Value 연산이 핵심입니다.",
            "문장에서 중요한 단어에 집중하는 방식이라는 정도는 알고 있습니다. Q, K, V 계산이나 multi-head attention은 정확히 설명하기 어렵습니다.",
            "Attention은 중요하지 않은 데이터를 삭제해서 GPU 메모리를 줄이는 기능이라고 알고 있습니다. 단어 길이를 계산하는 값에 가깝다고 생각합니다.",
            "시퀀스 길이에 따라 계산량이 커지는 한계가 있어 긴 컨텍스트에서는 sparse attention, chunking, retrieval 같은 대안을 함께 고려합니다.",
        ),
    ),
    (
        ("Precision", "Recall"),
        TopicProfile(
            "Precision과 Recall",
            "Precision은 양성으로 예측한 것 중 실제 양성 비율이고, Recall은 실제 양성 중 모델이 찾아낸 비율입니다.",
            "둘 다 분류 모델 평가 지표라는 정도는 알고 있습니다. 어떤 상황에서 하나를 더 중요하게 봐야 하는지는 아직 헷갈립니다.",
            "Precision은 모델 속도이고 Recall은 데이터 개수라고 알고 있습니다. 둘 다 높으면 좋지만 정확도와 거의 같은 지표라고 생각합니다.",
            "스팸 탐지, 질병 진단처럼 비용 구조에 따라 임계값을 조정해야 하며 F1, PR curve, confusion matrix를 함께 봐야 합니다.",
        ),
    ),
    (
        ("Docker",),
        TopicProfile(
            "Docker",
            "애플리케이션과 실행 환경을 이미지로 패키징해 컨테이너로 실행하는 기술입니다. VM보다 가볍고 배포 환경 일관성을 높일 수 있습니다.",
            "환경을 묶어서 실행하는 도구라는 정도는 알고 있습니다. 이미지 레이어나 네트워크, 볼륨 관리까지는 자세히 설명하기 어렵습니다.",
            "Docker는 작은 가상머신이라 운영체제를 매번 통째로 띄우는 방식이라고 알고 있습니다. 컨테이너 안 데이터는 항상 안전하게 보존된다고 생각합니다.",
            "이미지 크기, 보안 취약점, root 권한, 볼륨 영속성, 네트워크 격리를 함께 관리해야 하며 운영에서는 재현성과 보안 스캔이 중요합니다.",
        ),
    ),
    (
        ("Linux", "프로세스"),
        TopicProfile(
            "Linux 프로세스 관리",
            "Linux에서는 ps, top, htop, pidstat 등으로 프로세스 상태와 CPU, 메모리 사용량을 확인하고 kill이나 systemctl로 종료와 재시작을 관리할 수 있습니다.",
            "ps나 top으로 프로세스를 본다는 정도는 알고 있습니다. 상태 값이나 시그널 종류, systemd 서비스 관리까지는 자세히 설명하기 어렵습니다.",
            "Linux 프로세스는 터미널 창 하나를 뜻한다고 알고 있습니다. 서버가 느리면 프로세스를 전부 kill하면 해결되고, 시그널 차이는 크게 중요하지 않다고 생각합니다.",
            "운영 중에는 원인 분석 없이 종료하면 장애가 커질 수 있어 로그, open file, thread, 메모리 누수, systemd 재시작 정책을 함께 확인해야 합니다.",
        ),
    ),
    (
        ("Kubernetes", "Pod"),
        TopicProfile(
            "Kubernetes Pod",
            "Pod는 Kubernetes에서 배포 가능한 가장 작은 단위로 하나 이상의 컨테이너가 네트워크와 볼륨을 공유합니다.",
            "컨테이너를 실행하는 단위라는 정도는 알고 있습니다. Pod와 Deployment, Service의 역할 차이는 아직 명확히 설명하기 어렵습니다.",
            "Pod는 물리 서버 한 대를 뜻하고, 컨테이너가 많아지면 Pod가 자동으로 DB를 복제해준다고 알고 있습니다.",
            "Pod는 일시적이므로 직접 의존하기보다 Deployment, Service, readiness probe, resource limit을 통해 운영 안정성을 확보해야 합니다.",
        ),
    ),
    (
        ("CI/CD",),
        TopicProfile(
            "CI/CD",
            "코드 변경을 자동으로 빌드, 테스트, 배포해 통합과 릴리즈 과정을 반복 가능하게 만드는 파이프라인입니다.",
            "자동으로 테스트하고 배포하는 흐름이라는 정도는 알고 있습니다. 어떤 단계에서 품질 게이트를 둬야 하는지는 구체적으로 설명하기 어렵습니다.",
            "CI/CD는 코드를 자동으로 잘 짜주는 도구라고 알고 있습니다. 테스트가 실패해도 배포만 빠르게 되면 좋은 파이프라인이라고 생각합니다.",
            "테스트 범위, 롤백 전략, secret 관리, 배포 승인, artifact 추적성을 넣어야 속도와 안정성을 같이 얻을 수 있습니다.",
        ),
    ),
    (
        ("로드 밸런서", "Load Balancer"),
        TopicProfile(
            "로드 밸런서",
            "여러 서버로 트래픽을 분산해 가용성과 처리량을 높이는 구성 요소입니다. 헬스 체크로 비정상 인스턴스를 제외할 수 있습니다.",
            "요청을 여러 서버로 나눠주는 장비라는 정도는 알고 있습니다. L4와 L7 차이나 헬스 체크 방식은 자세히 설명하기 어렵습니다.",
            "로드 밸런서는 서버 CPU를 직접 늘려주는 기능이라고 알고 있습니다. 한 대만 두면 모든 장애가 자동으로 해결된다고 생각합니다.",
            "세션 유지, TLS 종료, 헬스 체크 기준, 알고리즘, 단일 장애점 제거를 함께 설계해야 실제 장애 상황에서도 효과가 있습니다.",
        ),
    ),
    (
        ("DNS",),
        TopicProfile(
            "DNS",
            "도메인 이름을 IP 주소로 변환하는 분산 네임 시스템입니다. 클라이언트는 캐시와 재귀/권한 DNS를 거쳐 레코드를 조회합니다.",
            "도메인을 IP로 바꿔주는 시스템이라는 정도는 알고 있습니다. TTL이나 레코드 종류, 조회 흐름은 깊게 설명하기 어렵습니다.",
            "DNS는 브라우저가 HTML 파일을 다운로드하는 프로토콜이라고 알고 있습니다. IP가 바뀌면 사용자가 새로고침하면 바로 반영된다고 생각합니다.",
            "TTL 때문에 변경 전파가 지연될 수 있고, 장애 대응에서는 다중 NS, 헬스 체크 기반 라우팅, 캐시 영향을 함께 고려해야 합니다.",
        ),
    ),
    (
        ("모니터링", "로깅"),
        TopicProfile(
            "모니터링과 로깅",
            "모니터링은 시스템 상태를 지표로 관측하고 로깅은 사건의 맥락을 기록해 장애 원인 분석을 돕습니다.",
            "장애를 확인하기 위해 필요하다는 정도는 알고 있습니다. 어떤 지표를 알림으로 잡아야 하는지나 로그 구조화는 자세히 모릅니다.",
            "모니터링은 서버 화면을 계속 보는 것이고 로깅은 printf를 많이 찍는 정도라고 생각합니다. 로그가 많을수록 항상 좋은 운영이라고 봅니다.",
            "SLI/SLO, 알림 피로도, trace id, 구조화 로그, 대시보드 기준을 잡아야 장애를 빨리 감지하고 원인을 좁힐 수 있습니다.",
        ),
    ),
]


ROLE_DEFAULTS = {
    "backend": TopicProfile(
        "백엔드 시스템 설계",
        "요청 처리, 데이터 정합성, 성능, 장애 대응을 함께 고려해 서버 기능을 안정적으로 구현하는 주제입니다.",
        "백엔드 개발에서 중요하다는 점은 알고 있지만 내부 동작이나 운영 기준까지는 자세히 설명하기 어렵습니다.",
        "서버가 알아서 처리하는 부분이라 개발자는 설정만 맞추면 된다고 생각합니다. 원리보다 빠르게 구현하는 게 더 중요하다고 봅니다.",
        "트래픽, 데이터 변경 패턴, 장애 시나리오, 관측 가능성을 함께 고려해야 운영 가능한 설계가 됩니다.",
    ),
    "frontend": TopicProfile(
        "프론트엔드 개발",
        "사용자 화면을 안정적으로 렌더링하고 상태, 성능, 접근성, 브라우저 동작을 고려해 구현하는 주제입니다.",
        "프론트엔드에서 자주 쓰는 개념이라는 정도는 알고 있지만 브라우저 내부 동작이나 성능 기준은 깊게 설명하기 어렵습니다.",
        "화면에 보이기만 하면 되는 부분이라 브라우저가 대부분 자동으로 최적화한다고 생각합니다. 세부 원리는 크게 중요하지 않다고 봅니다.",
        "사용자 경험, 렌더링 비용, 번들 크기, 접근성, 운영 환경 디버깅까지 함께 고려해야 합니다.",
    ),
    "data_ai": TopicProfile(
        "데이터·AI/ML",
        "데이터 품질, 모델 학습, 평가 지표, 배포 후 모니터링을 함께 고려해야 하는 머신러닝 주제입니다.",
        "모델 성능과 관련된 개념이라는 점은 알고 있지만 수식, 지표 해석, 운영 적용 기준은 아직 부족합니다.",
        "모델이 데이터를 많이 보면 자동으로 좋아지기 때문에 전처리나 평가 지표는 크게 중요하지 않다고 생각합니다.",
        "학습 데이터 분포, 평가 지표, 재현성, 드리프트, 운영 비용을 함께 봐야 실서비스에서 신뢰할 수 있습니다.",
    ),
    "devops": TopicProfile(
        "인프라·DevOps",
        "배포, 운영, 관측, 장애 대응을 자동화하고 안정적인 서비스 환경을 유지하기 위한 주제입니다.",
        "운영 자동화와 안정성에 필요하다는 점은 알고 있지만 세부 구성이나 장애 대응 기준은 자세히 설명하기 어렵습니다.",
        "서버는 한 번 설정하면 거의 자동으로 돌아가므로 장애가 나면 재시작하면 된다고 생각합니다. 문서화나 모니터링은 부가적인 일입니다.",
        "가용성, 보안, 롤백, 비용, 알림 기준을 함께 설계해야 운영 중 예측 가능한 대응이 가능합니다.",
    ),
}


ATTITUDE_PROFILE = TopicProfile(
    "IT 협업 상황",
    "먼저 사실과 의견을 분리하고 영향도, 일정, 사용자 관점, 기술적 근거를 공유한 뒤 팀 기준에 맞춰 결정해야 합니다.",
    "상대 의견을 듣고 팀 기준에 맞춰 결정해야 한다는 점은 알고 있습니다. 다만 어떤 근거를 수집하고 어떻게 합의할지는 구체성이 부족합니다.",
    "일단 제 방식대로 빠르게 진행하고 문제가 생기면 나중에 고치면 된다고 생각합니다. 회의가 길어지면 개발 속도만 늦어진다고 봅니다.",
    "단기 대응과 장기 개선을 분리하고, 담당자와 검증 기준을 남기며 회고를 통해 재발 방지 액션까지 추적해야 합니다.",
)


QUESTION_VARIANTS = [
    "{base}",
    "{stem} 핵심 원리와 필요한 이유를 설명해주세요.",
    "{stem} 실무에서 주의할 점을 포함해 설명해주세요.",
    "{stem} 장점과 한계를 함께 설명해주세요.",
    "{stem} 문제가 발생했을 때 어떻게 확인할지 설명해주세요.",
]


def parse_prompt(path: Path) -> tuple[str, str, list[SeedQuestion]]:
    text = path.read_text(encoding="utf-8")
    score_band = re.search(r"점수 기준:\s*(\d+-\d+)점", text).group(1)
    role_key = path.stem.rsplit("_", 1)[0]

    seeds: list[SeedQuestion] = []
    in_seed = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "## 참고 시드 질문":
            in_seed = True
            continue
        if in_seed and line.startswith("## 출력 형식"):
            break
        match = re.match(r"^\d+\.\s*\[([^\]]+)\]\s*(.+)$", line)
        if in_seed and match:
            seeds.append(SeedQuestion(match.group(1), match.group(2).strip()))

    if not seeds:
        raise ValueError(f"No seed questions in {path}")
    return role_key, score_band, seeds


def stem_question(question: str) -> str:
    stem = re.sub(r"(설명해주세요\.?|말해주세요\.?)$", "", question).strip()
    stem = stem.rstrip(".?")
    return stem


def make_question(seed: SeedQuestion, index: int) -> str:
    if index < len_seed_cycle_limit(seed):
        return seed.question
    stem = stem_question(seed.question)
    template = QUESTION_VARIANTS[index % len(QUESTION_VARIANTS)]
    return template.format(base=seed.question, stem=stem)


def len_seed_cycle_limit(seed: SeedQuestion) -> int:
    # Keep the first full pass identical to the original seed pool.
    return 10_000 if seed.question_type else 0


def profile_for(role_key: str, seed: SeedQuestion, question: str) -> TopicProfile:
    normalized = question.lower()
    for keywords, profile in PROFILES:
        if all(has_keyword(normalized, keyword) for keyword in keywords):
            return profile

    if seed.question_type == "IT 맥락 인성·사고방식":
        return ATTITUDE_PROFILE
    return ROLE_DEFAULTS[role_key]


def has_keyword(normalized_question: str, keyword: str) -> bool:
    normalized_keyword = keyword.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9+.#/-]*", normalized_keyword):
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])", normalized_question) is not None
    return normalized_keyword in normalized_question


def score_for(score_band: str, index: int) -> int:
    lo, hi = (int(x) for x in score_band.split("-"))
    return lo + (index % (hi - lo + 1))


def make_answer(profile: TopicProfile, score_band: str, score: int, seed: SeedQuestion, index: int) -> str:
    concept = extract_concept(seed.question)
    if profile in ROLE_DEFAULTS.values():
        profile = specialize_default(profile, concept)

    if score_band == "1-3":
        answer = profile.wrong
        if score == 1:
            answer += " 정확한 예시나 근거는 잘 떠오르지 않습니다."
        elif score == 2:
            answer += " 비슷한 용어들과도 조금 헷갈리지만 큰 차이는 없다고 생각합니다."
        else:
            answer += " 기본 개념은 들어봤지만 실제로 어떻게 동작하는지는 아직 잘 모르겠습니다."
    elif score_band == "4-6":
        answer = profile.shallow
        if score == 4:
            answer += " 그래서 면접에서 더 깊게 질문을 받으면 답변이 막힐 수 있을 것 같습니다."
        elif score == 5:
            answer += " 개념 정의는 말할 수 있지만 실제 장애나 성능 문제와 연결하는 부분은 부족합니다."
        else:
            answer += " 간단한 사용 목적은 설명할 수 있지만 구체적인 예시나 트레이드오프는 더 공부해야 합니다."
    elif score_band == "7-8":
        answer = profile.correct
        if score == 7:
            answer += f" 실무에서는 {profile.tradeoff}"
        else:
            answer += f" 또한 {profile.tradeoff} 간단한 예시나 주의점까지 연결해 설명할 수 있습니다."
    else:
        answer = profile.correct
        answer += f" 여기서 중요한 점은 {profile.tradeoff}"
        answer += f" {advanced_detail(profile.name, concept, score, index)}"
        if seed.question_type == "IT 맥락 인성·사고방식":
            answer += " 특히 면접에서는 감정적인 태도보다 영향도 판단, 커뮤니케이션, 담당자 지정, 후속 개선 추적까지 함께 말할 수 있어야 합니다."
        else:
            answer += " 면접에서는 단순 정의보다 선택 근거, 실패 시 관측 지표, 대안 비교까지 말하는 수준이 좋습니다."

    # Add small variation without changing quality band.
    if index % 4 == 1 and score_band in {"7-8", "9-10"}:
        answer += " 운영 관점에서는 로그와 지표로 검증 가능한 형태로 설명하는 것이 중요합니다."
    elif index % 4 == 2 and score_band in {"4-6", "7-8"}:
        answer += " 다만 아직 세부 구현까지 완전히 익숙한 수준은 아닙니다." if score_band == "4-6" else " 팀 상황에 따라 과한 설계가 되지 않도록 범위를 조절해야 합니다."

    return fit_length(answer, score_band)


def advanced_detail(profile_name: str, concept: str, score: int, index: int) -> str:
    subject = profile_name if profile_name else concept
    examples = [
        f"예를 들어 {subject}을/를 설명할 때는 정상 흐름뿐 아니라 장애, 재시도, 동시 요청 상황에서 어떤 문제가 생기는지까지 가정해야 합니다.",
        f"운영에서는 {subject}의 장점만 말하기보다 로그, 메트릭, 알림 기준을 통해 실제 효과를 검증할 수 있는지가 더 중요합니다.",
        f"{subject}을/를 설명할 때는 단일 정답처럼 말하기보다 트래픽 규모, 데이터 변경 빈도, 팀의 운영 역량에 따라 선택이 달라질 수 있음을 보여줘야 합니다.",
        f"실제 프로젝트라면 작은 범위에서 먼저 적용하고, 응답 시간이나 오류율 같은 지표를 비교한 뒤 확대하는 접근이 안전합니다.",
    ]
    detail = examples[index % len(examples)]
    if score == 10:
        detail += " 10점 답변이라면 여기서 한 단계 더 나아가 엣지 케이스와 롤백 전략까지 함께 제시하는 것이 자연스럽습니다."
    return detail


def extract_concept(question: str) -> str:
    concept = re.sub(r"\([^)]*\)", "", question)
    concept = re.sub(r"(에 대해|이 무엇|가 무엇|을 사용하는 이유|를 사용하는 이유|의 차이|가 필요한 이유|이 필요한 이유).*$", "", concept)
    concept = re.sub(r"(설명하고|설명해주세요|말해주세요|장단점.*|방법.*).*$", "", concept)
    concept = concept.strip(" .?")
    return concept[:60] or "해당 개념"


def specialize_default(profile: TopicProfile, concept: str) -> TopicProfile:
    return TopicProfile(
        profile.name,
        f"{concept}은/는 {profile.correct}",
        f"{concept}은/는 {profile.shallow}",
        f"{concept}은/는 {profile.wrong}",
        profile.tradeoff,
    )


def fit_length(answer: str, score_band: str) -> str:
    fillers = {
        "1-3": [
            " 그래서 정확한 원리나 예외 상황은 잘 설명하기 어렵습니다.",
            " 면접에서 추가 질문을 받으면 개념 구분이 흔들릴 수 있습니다.",
        ],
        "4-6": [
            " 구체적인 판단 기준과 실무 경험은 아직 부족합니다.",
            " 왜 그런 선택을 하는지까지는 설명이 충분하지 않습니다.",
        ],
        "7-8": [
            " 적용 전후 효과를 확인하는 과정도 필요합니다.",
            " 다만 실제 수치나 장애 사례까지 제시하면 더 좋아집니다.",
        ],
        "9-10": [
            " 또한 장애 대응과 재발 방지까지 고려해야 합니다.",
            " 이런 관점은 실제 운영 경험과 깊은 이해를 보여줍니다.",
        ],
    }[score_band]
    i = 0
    while len(answer) < MIN_LENGTH[score_band]:
        answer += fillers[i % len(fillers)]
        i += 1
    if len(answer) > 700:
        answer = answer[:697].rstrip() + "..."
    return answer


def generate_file(prompt_path: Path) -> int:
    role_key, score_band, seeds = parse_prompt(prompt_path)
    target = TARGETS_PER_BAND[role_key]
    rows = []
    for index in range(target):
        seed = seeds[index % len(seeds)]
        # Keep original seed questions during the first pass, then use variants.
        question = seed.question if index < len(seeds) else make_variant(seed.question, index)
        profile = profile_for(role_key, seed, question)
        score = score_for(score_band, index)
        rows.append(
            {
                "question": question,
                "answer": make_answer(profile, score_band, score, seed, index),
                "score": score,
            }
        )

    output_path = RESPONSES_DIR / f"{prompt_path.stem}.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def make_variant(question: str, index: int) -> str:
    stem = stem_question(question)
    template = QUESTION_VARIANTS[index % len(QUESTION_VARIANTS)]
    variant = template.format(base=question, stem=stem)
    return re.sub(r"\s+", " ", variant).strip()


def main() -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for prompt_path in sorted(PROMPTS_DIR.glob("*.txt")):
        count = generate_file(prompt_path)
        total += count
        print(f"{prompt_path.stem}.json: {count}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
