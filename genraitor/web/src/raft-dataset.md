---
theme: dashboard
title: RAFT Dataset
toc: true
sql:
  raft: ./data/raft.parquet
---

```sql id=raft
SELECT
  *
FROM raft
```

# Explore the RAFT dataset
```js
const selection = view(Inputs.table(raft, {multiple: false, columns: ['question'], required: true}));


function display_raft(row, i) {
  if(row === null){
    return html`<div>Make a selection from the table above to view the data.</div>`
  }
  return html`
  <div class="card">
      <div><b>${row.question}</b></div>
      <h2>Answer</h2>
      <div><i>${row.cot_answer}</i></div>
      <h2>Context</h2>
      <div><i>${row.context}</i></div>
      <h2>Instruction</h2>
      <div><i>${row.instruction}</i></div>
      <h2>Oracle Context</h2>
      <div><i>${row.oracle_context}</i></div>
  </div>
  `;
}
```

${[selection].map(display_raft)}

# Another title

## Subtitle

# One more
