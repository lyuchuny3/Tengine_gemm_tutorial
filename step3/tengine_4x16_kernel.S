/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/

// x0: C
// x1: B
// x2: A
// x3: k

    .section .text,"ax"
    .align 5
    .type tengine_4x16_kernel STT_FUNC
    .global tengine_4x16_kernel
    
tengine_4x16_kernel:

	sub	sp, sp, #0x20
	stp	d8, d9,  [sp]
	stp	d10,d11, [sp, 0x10]

    lsr	x6, x3, 0x2               //x6 = k/4

	movi	d16, #0
	movi	d17, #0
	movi	d18, #0
	movi	d19, #0
	movi	d20, #0
	movi	d21, #0
	movi	d22, #0
	movi	d23, #0
	movi	d24, #0
	movi	d25, #0
	movi	d26, #0
	movi	d27, #0
	movi	d28, #0
	movi	d29, #0
	movi	d30, #0
	movi	d31, #0

loop4:  
	ldr	q0, [x1]	
	ldr	q1, [x1, 0x10]	
	ldp	q2, q3, [x1, 0x20]

	ldp	q4, q5, [x2]
	ldp	q6, q7, [x2, 0x20]		
	ldp	q8, q9, [x2, 0x40]
	ldp	q10,q11,[x2, 0x60]

	fmla	v16.4s, v0.4s,  v4.s[0]
	fmla	v17.4s, v0.4s,  v4.s[1]
	fmla	v18.4s, v0.4s,  v4.s[2]
	fmla	v19.4s, v0.4s,  v4.s[3]
	fmla	v20.4s, v0.4s,  v5.s[0]
	fmla	v21.4s, v0.4s,  v5.s[1]
	fmla	v22.4s, v0.4s,  v5.s[2]
	fmla	v23.4s, v0.4s,  v5.s[3]
	fmla	v24.4s, v0.4s,  v6.s[0]
	fmla	v25.4s, v0.4s,  v6.s[1]
	fmla	v26.4s, v0.4s,  v6.s[2]
	fmla	v27.4s, v0.4s,  v6.s[3]
	fmla	v28.4s, v0.4s,  v7.s[0]
	fmla	v29.4s, v0.4s,  v7.s[1]
	fmla	v30.4s, v0.4s,  v7.s[2]
	fmla	v31.4s, v0.4s,  v7.s[3]
	
	fmla	v16.4s, v1.4s,  v8.s[0]
	fmla	v17.4s, v1.4s,  v8.s[1]
	fmla	v18.4s, v1.4s,  v8.s[2]
	fmla	v19.4s, v1.4s,  v8.s[3]
	fmla	v20.4s, v1.4s,  v9.s[0]
	fmla	v21.4s, v1.4s,  v9.s[1]
	fmla	v22.4s, v1.4s,  v9.s[2]
	fmla	v23.4s, v1.4s,  v9.s[3]
	fmla	v24.4s, v1.4s, v10.s[0]
	fmla	v25.4s, v1.4s, v10.s[1]
	fmla	v26.4s, v1.4s, v10.s[2]
	fmla	v27.4s, v1.4s, v10.s[3]
	subs	x6, x6, #0x1
	prfm	pldl1keep, [x1, 0x80]
	fmla	v28.4s, v1.4s, v11.s[0]
	fmla	v29.4s, v1.4s, v11.s[1]
	fmla	v30.4s, v1.4s, v11.s[2]
	fmla	v31.4s, v1.4s, v11.s[3]

	ldp	q4, q5, [x2, 0x80] 
	ldp	q6, q7, [x2, 0xa0]
	ldp	q8, q9, [x2, 0xc0] 
	ldp	q10,q11,[x2, 0xe0]

	fmla	v16.4s, v2.4s,  v4.s[0]
	fmla	v17.4s, v2.4s,  v4.s[1]
	fmla	v18.4s, v2.4s,  v4.s[2]
	fmla	v19.4s, v2.4s,  v4.s[3]
	add	x1, x1, #0x40
	fmla	v20.4s, v2.4s,  v5.s[0]
	fmla	v21.4s, v2.4s,  v5.s[1]
	fmla	v22.4s, v2.4s,  v5.s[2]
	fmla	v23.4s, v2.4s,  v5.s[3]
	fmla	v24.4s, v2.4s,  v6.s[0]
	fmla	v25.4s, v2.4s,  v6.s[1]
	fmla	v26.4s, v2.4s,  v6.s[2]
	fmla	v27.4s, v2.4s,  v6.s[3]
	prfm	pldl1keep, [x2, 0x140]
	fmla	v28.4s, v2.4s,  v7.s[0]
	fmla	v29.4s, v2.4s,  v7.s[1]
	fmla	v30.4s, v2.4s,  v7.s[2]
	fmla	v31.4s, v2.4s,  v7.s[3]

	fmla	v16.4s, v3.4s,  v8.s[0]
	fmla	v17.4s, v3.4s,  v8.s[1]
	fmla	v18.4s, v3.4s,  v8.s[2]
	fmla	v19.4s, v3.4s,  v8.s[3]
	prfm	pldl1keep, [x2, 0x180]
	fmla	v20.4s, v3.4s,  v9.s[0]
	fmla	v21.4s, v3.4s,  v9.s[1]
	fmla	v22.4s, v3.4s,  v9.s[2]
	fmla	v23.4s, v3.4s,  v9.s[3]
	prfm	pldl1keep, [x2, 0x1c0]
	fmla	v24.4s, v3.4s,  v10.s[0]
	fmla	v25.4s, v3.4s,  v10.s[1]
	fmla	v26.4s, v3.4s,  v10.s[2]
	fmla	v27.4s, v3.4s,  v10.s[3]
	prfm	pldl1keep, [x2, 0x200]
	add	x2, x2, #0x100
	fmla	v28.4s, v3.4s,  v11.s[0]
	fmla	v29.4s, v3.4s,  v11.s[1]
	fmla	v30.4s, v3.4s,  v11.s[2]
	fmla	v31.4s, v3.4s,  v11.s[3]
	b.ne	loop4


finish:
	stp     q16, q17 ,[x0]
	stp     q18, q19 ,[x0, #0x20]
	stp     q20, q21 ,[x0, #0x40]
	stp     q22, q23 ,[x0, #0x60]
	stp     q24, q25 ,[x0, #0x80]
	stp     q26, q27 ,[x0, #0xa0]
	stp     q28, q29 ,[x0, #0xc0]
	stp     q30, q31 ,[x0, #0xe0]

	ldp     d8, d9, [sp]
	ldp     d10,d11,[sp, 0x10]
	add     sp, sp, 0x20
	
	ret
	.space  256
	.end