
./build/kernel:     file format elf64-x86-64


Disassembly of section .init:

0000000000401008 <_init>:
  401008:	48 83 ec 08          	sub    $0x8,%rsp
  40100c:	48 8b 05 e5 3f 20 00 	mov    0x203fe5(%rip),%rax        # 604ff8 <__gmon_start__>
  401013:	48 85 c0             	test   %rax,%rax
  401016:	74 05                	je     40101d <_init+0x15>
  401018:	e8 93 00 00 00       	callq  4010b0 <__gmon_start__@plt>
  40101d:	48 83 c4 08          	add    $0x8,%rsp
  401021:	c3                   	retq   

Disassembly of section .plt:

0000000000401030 <.plt>:
  401030:	ff 35 d2 3f 20 00    	pushq  0x203fd2(%rip)        # 605008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401036:	ff 25 d4 3f 20 00    	jmpq   *0x203fd4(%rip)        # 605010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40103c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401040 <__cxa_begin_catch@plt>:
  401040:	ff 25 d2 3f 20 00    	jmpq   *0x203fd2(%rip)        # 605018 <__cxa_begin_catch@CXXABI_1.3>
  401046:	68 00 00 00 00       	pushq  $0x0
  40104b:	e9 e0 ff ff ff       	jmpq   401030 <.plt>

0000000000401050 <memcpy@plt>:
  401050:	ff 25 ca 3f 20 00    	jmpq   *0x203fca(%rip)        # 605020 <memcpy@GLIBC_2.14>
  401056:	68 01 00 00 00       	pushq  $0x1
  40105b:	e9 d0 ff ff ff       	jmpq   401030 <.plt>

0000000000401060 <_ZN2cv3Mat8copySizeERKS0_@plt>:
  401060:	ff 25 c2 3f 20 00    	jmpq   *0x203fc2(%rip)        # 605028 <_ZN2cv3Mat8copySizeERKS0_>
  401066:	68 02 00 00 00       	pushq  $0x2
  40106b:	e9 c0 ff ff ff       	jmpq   401030 <.plt>

0000000000401070 <_ZSt17__throw_bad_allocv@plt>:
  401070:	ff 25 ba 3f 20 00    	jmpq   *0x203fba(%rip)        # 605030 <_ZSt17__throw_bad_allocv@GLIBCXX_3.4>
  401076:	68 03 00 00 00       	pushq  $0x3
  40107b:	e9 b0 ff ff ff       	jmpq   401030 <.plt>

0000000000401080 <_Znwm@plt>:
  401080:	ff 25 b2 3f 20 00    	jmpq   *0x203fb2(%rip)        # 605038 <_Znwm@GLIBCXX_3.4>
  401086:	68 04 00 00 00       	pushq  $0x4
  40108b:	e9 a0 ff ff ff       	jmpq   401030 <.plt>

0000000000401090 <free@plt>:
  401090:	ff 25 aa 3f 20 00    	jmpq   *0x203faa(%rip)        # 605040 <free@GLIBC_2.2.5>
  401096:	68 05 00 00 00       	pushq  $0x5
  40109b:	e9 90 ff ff ff       	jmpq   401030 <.plt>

00000000004010a0 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>:
  4010a0:	ff 25 a2 3f 20 00    	jmpq   *0x203fa2(%rip)        # 605048 <_ZN2cv12_OutputArrayC1ERNS_3MatE>
  4010a6:	68 06 00 00 00       	pushq  $0x6
  4010ab:	e9 80 ff ff ff       	jmpq   401030 <.plt>

00000000004010b0 <__gmon_start__@plt>:
  4010b0:	ff 25 9a 3f 20 00    	jmpq   *0x203f9a(%rip)        # 605050 <__gmon_start__>
  4010b6:	68 07 00 00 00       	pushq  $0x7
  4010bb:	e9 70 ff ff ff       	jmpq   401030 <.plt>

00000000004010c0 <_ZN2cv3Mat10deallocateEv@plt>:
  4010c0:	ff 25 92 3f 20 00    	jmpq   *0x203f92(%rip)        # 605058 <_ZN2cv3Mat10deallocateEv>
  4010c6:	68 08 00 00 00       	pushq  $0x8
  4010cb:	e9 60 ff ff ff       	jmpq   401030 <.plt>

00000000004010d0 <_ZdlPv@plt>:
  4010d0:	ff 25 8a 3f 20 00    	jmpq   *0x203f8a(%rip)        # 605060 <_ZdlPv@GLIBCXX_3.4>
  4010d6:	68 09 00 00 00       	pushq  $0x9
  4010db:	e9 50 ff ff ff       	jmpq   401030 <.plt>

00000000004010e0 <__gxx_personality_v0@plt>:
  4010e0:	ff 25 82 3f 20 00    	jmpq   *0x203f82(%rip)        # 605068 <__gxx_personality_v0@CXXABI_1.3>
  4010e6:	68 0a 00 00 00       	pushq  $0xa
  4010eb:	e9 40 ff ff ff       	jmpq   401030 <.plt>

00000000004010f0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>:
  4010f0:	ff 25 7a 3f 20 00    	jmpq   *0x203f7a(%rip)        # 605070 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE>
  4010f6:	68 0b 00 00 00       	pushq  $0xb
  4010fb:	e9 30 ff ff ff       	jmpq   401030 <.plt>

0000000000401100 <_Unwind_Resume@plt>:
  401100:	ff 25 72 3f 20 00    	jmpq   *0x203f72(%rip)        # 605078 <_Unwind_Resume@GCC_3.0>
  401106:	68 0c 00 00 00       	pushq  $0xc
  40110b:	e9 20 ff ff ff       	jmpq   401030 <.plt>

0000000000401110 <_ZNSt8ios_base4InitD1Ev@plt>:
  401110:	ff 25 6a 3f 20 00    	jmpq   *0x203f6a(%rip)        # 605080 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  401116:	68 0d 00 00 00       	pushq  $0xd
  40111b:	e9 10 ff ff ff       	jmpq   401030 <.plt>

0000000000401120 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>:
  401120:	ff 25 62 3f 20 00    	jmpq   *0x203f62(%rip)        # 605088 <_ZN2cv11_InputArrayC1ERKNS_3MatE>
  401126:	68 0e 00 00 00       	pushq  $0xe
  40112b:	e9 00 ff ff ff       	jmpq   401030 <.plt>

0000000000401130 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>:
  401130:	ff 25 5a 3f 20 00    	jmpq   *0x203f5a(%rip)        # 605090 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE>
  401136:	68 0f 00 00 00       	pushq  $0xf
  40113b:	e9 f0 fe ff ff       	jmpq   401030 <.plt>

0000000000401140 <__cxa_end_catch@plt>:
  401140:	ff 25 52 3f 20 00    	jmpq   *0x203f52(%rip)        # 605098 <__cxa_end_catch@CXXABI_1.3>
  401146:	68 10 00 00 00       	pushq  $0x10
  40114b:	e9 e0 fe ff ff       	jmpq   401030 <.plt>

0000000000401150 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>:
  401150:	ff 25 4a 3f 20 00    	jmpq   *0x203f4a(%rip)        # 6050a0 <_ZNSs4_Rep10_M_disposeERKSaIcE@GLIBCXX_3.4>
  401156:	68 11 00 00 00       	pushq  $0x11
  40115b:	e9 d0 fe ff ff       	jmpq   401030 <.plt>

0000000000401160 <__cxa_rethrow@plt>:
  401160:	ff 25 42 3f 20 00    	jmpq   *0x203f42(%rip)        # 6050a8 <__cxa_rethrow@CXXABI_1.3>
  401166:	68 12 00 00 00       	pushq  $0x12
  40116b:	e9 c0 fe ff ff       	jmpq   401030 <.plt>

0000000000401170 <calloc@plt>:
  401170:	ff 25 3a 3f 20 00    	jmpq   *0x203f3a(%rip)        # 6050b0 <calloc@GLIBC_2.2.5>
  401176:	68 13 00 00 00       	pushq  $0x13
  40117b:	e9 b0 fe ff ff       	jmpq   401030 <.plt>

0000000000401180 <__cxa_atexit@plt>:
  401180:	ff 25 32 3f 20 00    	jmpq   *0x203f32(%rip)        # 6050b8 <__cxa_atexit@GLIBC_2.2.5>
  401186:	68 14 00 00 00       	pushq  $0x14
  40118b:	e9 a0 fe ff ff       	jmpq   401030 <.plt>

0000000000401190 <_ZN2cv8fastFreeEPv@plt>:
  401190:	ff 25 2a 3f 20 00    	jmpq   *0x203f2a(%rip)        # 6050c0 <_ZN2cv8fastFreeEPv>
  401196:	68 15 00 00 00       	pushq  $0x15
  40119b:	e9 90 fe ff ff       	jmpq   401030 <.plt>

00000000004011a0 <__pthread_key_create@plt>:
  4011a0:	ff 25 22 3f 20 00    	jmpq   *0x203f22(%rip)        # 6050c8 <__pthread_key_create>
  4011a6:	68 16 00 00 00       	pushq  $0x16
  4011ab:	e9 80 fe ff ff       	jmpq   401030 <.plt>

00000000004011b0 <__libc_start_main@plt>:
  4011b0:	ff 25 1a 3f 20 00    	jmpq   *0x203f1a(%rip)        # 6050d0 <__libc_start_main@GLIBC_2.2.5>
  4011b6:	68 17 00 00 00       	pushq  $0x17
  4011bb:	e9 70 fe ff ff       	jmpq   401030 <.plt>

00000000004011c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>:
  4011c0:	ff 25 12 3f 20 00    	jmpq   *0x203f12(%rip)        # 6050d8 <_ZNSs4_Rep10_M_destroyERKSaIcE@GLIBCXX_3.4>
  4011c6:	68 18 00 00 00       	pushq  $0x18
  4011cb:	e9 60 fe ff ff       	jmpq   401030 <.plt>

00000000004011d0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4011d0:	ff 25 0a 3f 20 00    	jmpq   *0x203f0a(%rip)        # 6050e0 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  4011d6:	68 19 00 00 00       	pushq  $0x19
  4011db:	e9 50 fe ff ff       	jmpq   401030 <.plt>

00000000004011e0 <_ZN2cv5splitERKNS_3MatEPS0_@plt>:
  4011e0:	ff 25 02 3f 20 00    	jmpq   *0x203f02(%rip)        # 6050e8 <_ZN2cv5splitERKNS_3MatEPS0_>
  4011e6:	68 1a 00 00 00       	pushq  $0x1a
  4011eb:	e9 40 fe ff ff       	jmpq   401030 <.plt>

00000000004011f0 <_ZN2cv6imreadERKSsi@plt>:
  4011f0:	ff 25 fa 3e 20 00    	jmpq   *0x203efa(%rip)        # 6050f0 <_ZN2cv6imreadERKSsi>
  4011f6:	68 1b 00 00 00       	pushq  $0x1b
  4011fb:	e9 30 fe ff ff       	jmpq   401030 <.plt>

0000000000401200 <_ZNSsC1EPKcRKSaIcE@plt>:
  401200:	ff 25 f2 3e 20 00    	jmpq   *0x203ef2(%rip)        # 6050f8 <_ZNSsC1EPKcRKSaIcE@GLIBCXX_3.4>
  401206:	68 1c 00 00 00       	pushq  $0x1c
  40120b:	e9 20 fe ff ff       	jmpq   401030 <.plt>

Disassembly of section .text:

0000000000401210 <main>:
  401210:	41 57                	push   %r15
  401212:	be 04 00 00 00       	mov    $0x4,%esi
  401217:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  40121c:	41 56                	push   %r14
  40121e:	41 55                	push   %r13
  401220:	41 54                	push   %r12
  401222:	55                   	push   %rbp
  401223:	53                   	push   %rbx
  401224:	48 81 ec 38 01 00 00 	sub    $0x138,%rsp
  40122b:	e8 40 ff ff ff       	callq  401170 <calloc@plt>
  401230:	be 04 00 00 00       	mov    $0x4,%esi
  401235:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  40123a:	49 89 c4             	mov    %rax,%r12
  40123d:	e8 2e ff ff ff       	callq  401170 <calloc@plt>
  401242:	be 04 00 00 00       	mov    $0x4,%esi
  401247:	bf 00 c0 12 00       	mov    $0x12c000,%edi
  40124c:	48 89 c5             	mov    %rax,%rbp
  40124f:	e8 1c ff ff ff       	callq  401170 <calloc@plt>
  401254:	be 04 00 00 00       	mov    $0x4,%esi
  401259:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  40125e:	48 89 c3             	mov    %rax,%rbx
  401261:	e8 0a ff ff ff       	callq  401170 <calloc@plt>
  401266:	be 04 00 00 00       	mov    $0x4,%esi
  40126b:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  401270:	49 89 c5             	mov    %rax,%r13
  401273:	e8 f8 fe ff ff       	callq  401170 <calloc@plt>
  401278:	be 04 00 00 00       	mov    $0x4,%esi
  40127d:	bf 00 b0 04 00       	mov    $0x4b000,%edi
  401282:	49 89 c6             	mov    %rax,%r14
  401285:	e8 e6 fe ff ff       	callq  401170 <calloc@plt>
  40128a:	49 89 c7             	mov    %rax,%r15
  40128d:	0f 31                	rdtsc  
  40128f:	4c 89 fa             	mov    %r15,%rdx
  401292:	4c 89 f6             	mov    %r14,%rsi
  401295:	4c 89 ef             	mov    %r13,%rdi
  401298:	e8 83 14 00 00       	callq  402720 <_Z11decodeImagePfS_S_>
  40129d:	0f 31                	rdtsc  
  40129f:	be 04 00 00 00       	mov    $0x4,%esi
  4012a4:	bf 40 00 00 00       	mov    $0x40,%edi
  4012a9:	e8 c2 fe ff ff       	callq  401170 <calloc@plt>
  4012ae:	48 89 c7             	mov    %rax,%rdi
  4012b1:	48 89 04 24          	mov    %rax,(%rsp)
  4012b5:	e8 56 05 00 00       	callq  401810 <_Z20generateCoefficientsPf>
  4012ba:	4c 8b 1c 24          	mov    (%rsp),%r11
  4012be:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
  4012c2:	45 31 c0             	xor    %r8d,%r8d
  4012c5:	be 67 66 66 66       	mov    $0x66666667,%esi
  4012ca:	c4 c1 78 10 23       	vmovups (%r11),%xmm4
  4012cf:	c5 f8 29 a4 24 f0 00 	vmovaps %xmm4,0xf0(%rsp)
  4012d6:	00 00 
  4012d8:	c4 c1 78 10 63 10    	vmovups 0x10(%r11),%xmm4
  4012de:	c5 f8 29 a4 24 d0 00 	vmovaps %xmm4,0xd0(%rsp)
  4012e5:	00 00 
  4012e7:	c4 c1 78 10 63 40    	vmovups 0x40(%r11),%xmm4
  4012ed:	c5 f8 29 a4 24 20 01 	vmovaps %xmm4,0x120(%rsp)
  4012f4:	00 00 
  4012f6:	c4 c1 78 10 63 50    	vmovups 0x50(%r11),%xmm4
  4012fc:	c5 f8 29 a4 24 b0 00 	vmovaps %xmm4,0xb0(%rsp)
  401303:	00 00 
  401305:	c4 c1 78 10 a3 80 00 	vmovups 0x80(%r11),%xmm4
  40130c:	00 00 
  40130e:	c5 f8 29 a4 24 e0 00 	vmovaps %xmm4,0xe0(%rsp)
  401315:	00 00 
  401317:	c4 c1 78 10 a3 90 00 	vmovups 0x90(%r11),%xmm4
  40131e:	00 00 
  401320:	c5 f8 29 a4 24 10 01 	vmovaps %xmm4,0x110(%rsp)
  401327:	00 00 
  401329:	c4 c1 78 10 a3 c0 00 	vmovups 0xc0(%r11),%xmm4
  401330:	00 00 
  401332:	c5 f8 29 64 24 20    	vmovaps %xmm4,0x20(%rsp)
  401338:	c4 c1 78 10 a3 d0 00 	vmovups 0xd0(%r11),%xmm4
  40133f:	00 00 
  401341:	c5 f8 29 64 24 30    	vmovaps %xmm4,0x30(%rsp)
  401347:	c4 c1 78 10 63 30    	vmovups 0x30(%r11),%xmm4
  40134d:	c5 f8 29 64 24 40    	vmovaps %xmm4,0x40(%rsp)
  401353:	c4 c1 78 10 63 20    	vmovups 0x20(%r11),%xmm4
  401359:	c5 f8 29 24 24       	vmovaps %xmm4,(%rsp)
  40135e:	c4 c1 78 10 63 60    	vmovups 0x60(%r11),%xmm4
  401364:	c5 f8 29 64 24 10    	vmovaps %xmm4,0x10(%rsp)
  40136a:	c4 c1 78 10 63 70    	vmovups 0x70(%r11),%xmm4
  401370:	c5 f8 29 64 24 50    	vmovaps %xmm4,0x50(%rsp)
  401376:	c4 c1 78 10 a3 a0 00 	vmovups 0xa0(%r11),%xmm4
  40137d:	00 00 
  40137f:	c5 f8 29 a4 24 a0 00 	vmovaps %xmm4,0xa0(%rsp)
  401386:	00 00 
  401388:	c4 c1 78 10 a3 b0 00 	vmovups 0xb0(%r11),%xmm4
  40138f:	00 00 
  401391:	c5 f8 29 a4 24 c0 00 	vmovaps %xmm4,0xc0(%rsp)
  401398:	00 00 
  40139a:	c4 c1 78 10 a3 e0 00 	vmovups 0xe0(%r11),%xmm4
  4013a1:	00 00 
  4013a3:	c5 f8 29 a4 24 00 01 	vmovaps %xmm4,0x100(%rsp)
  4013aa:	00 00 
  4013ac:	c4 c1 78 10 a3 f0 00 	vmovups 0xf0(%r11),%xmm4
  4013b3:	00 00 
  4013b5:	c5 f8 29 a4 24 90 00 	vmovaps %xmm4,0x90(%rsp)
  4013bc:	00 00 
  4013be:	66 90                	xchg   %ax,%ax
  4013c0:	44 89 c0             	mov    %r8d,%eax
  4013c3:	f7 ee                	imul   %esi
  4013c5:	44 89 c0             	mov    %r8d,%eax
  4013c8:	c1 f8 1f             	sar    $0x1f,%eax
  4013cb:	89 d1                	mov    %edx,%ecx
  4013cd:	c1 f9 08             	sar    $0x8,%ecx
  4013d0:	29 c1                	sub    %eax,%ecx
  4013d2:	8d 3c 8d 00 00 00 00 	lea    0x0(,%rcx,4),%edi
  4013d9:	89 c8                	mov    %ecx,%eax
  4013db:	c1 e0 04             	shl    $0x4,%eax
  4013de:	8d 14 0f             	lea    (%rdi,%rcx,1),%edx
  4013e1:	41 89 c1             	mov    %eax,%r9d
  4013e4:	44 89 c1             	mov    %r8d,%ecx
  4013e7:	41 01 f9             	add    %edi,%r9d
  4013ea:	c1 e2 07             	shl    $0x7,%edx
  4013ed:	41 c1 e1 08          	shl    $0x8,%r9d
  4013f1:	29 d1                	sub    %edx,%ecx
  4013f3:	0f 31                	rdtsc  
  4013f5:	41 8d 04 49          	lea    (%r9,%rcx,2),%eax
  4013f9:	41 c1 f9 02          	sar    $0x2,%r9d
  4013fd:	44 01 c9             	add    %r9d,%ecx
  401400:	c5 f8 28 84 24 f0 00 	vmovaps 0xf0(%rsp),%xmm0
  401407:	00 00 
  401409:	48 63 c9             	movslq %ecx,%rcx
  40140c:	48 98                	cltq   
  40140e:	48 c1 e1 02          	shl    $0x2,%rcx
  401412:	c5 78 28 94 24 d0 00 	vmovaps 0xd0(%rsp),%xmm10
  401419:	00 00 
  40141b:	49 8d 3c 0f          	lea    (%r15,%rcx,1),%rdi
  40141f:	4d 8d 0c 0e          	lea    (%r14,%rcx,1),%r9
  401423:	4c 01 e9             	add    %r13,%rcx
  401426:	c4 e2 79 18 19       	vbroadcastss (%rcx),%xmm3
  40142b:	48 c1 e0 02          	shl    $0x2,%rax
  40142f:	c4 41 78 28 e2       	vmovaps %xmm10,%xmm12
  401434:	4c 8d 14 03          	lea    (%rbx,%rax,1),%r10
  401438:	c4 c2 79 18 21       	vbroadcastss (%r9),%xmm4
  40143d:	48 8d 54 05 00       	lea    0x0(%rbp,%rax,1),%rdx
  401442:	c5 78 28 fb          	vmovaps %xmm3,%xmm15
  401446:	4c 01 e0             	add    %r12,%rax
  401449:	c4 e2 79 18 37       	vbroadcastss (%rdi),%xmm6
  40144e:	c5 78 28 f4          	vmovaps %xmm4,%xmm14
  401452:	c4 62 51 98 f8       	vfmadd132ps %xmm0,%xmm5,%xmm15
  401457:	c4 41 78 28 da       	vmovaps %xmm10,%xmm11
  40145c:	c5 f8 29 5c 24 60    	vmovaps %xmm3,0x60(%rsp)
  401462:	c5 f8 28 fe          	vmovaps %xmm6,%xmm7
  401466:	c4 62 51 98 f0       	vfmadd132ps %xmm0,%xmm5,%xmm14
  40146b:	c5 f8 29 b4 24 80 00 	vmovaps %xmm6,0x80(%rsp)
  401472:	00 00 
  401474:	c4 e2 51 98 f8       	vfmadd132ps %xmm0,%xmm5,%xmm7
  401479:	c5 78 28 ef          	vmovaps %xmm7,%xmm13
  40147d:	c5 f8 29 64 24 70    	vmovaps %xmm4,0x70(%rsp)
  401483:	c5 f8 28 9c 24 b0 00 	vmovaps 0xb0(%rsp),%xmm3
  40148a:	00 00 
  40148c:	c4 e2 79 18 51 04    	vbroadcastss 0x4(%rcx),%xmm2
  401492:	c4 c2 79 18 49 04    	vbroadcastss 0x4(%r9),%xmm1
  401498:	c4 62 51 98 e2       	vfmadd132ps %xmm2,%xmm5,%xmm12
  40149d:	c4 62 69 b8 e3       	vfmadd231ps %xmm3,%xmm2,%xmm12
  4014a2:	c4 e2 79 18 47 04    	vbroadcastss 0x4(%rdi),%xmm0
  4014a8:	c4 62 51 98 d9       	vfmadd132ps %xmm1,%xmm5,%xmm11
  4014ad:	c4 62 71 b8 db       	vfmadd231ps %xmm3,%xmm1,%xmm11
  4014b2:	c4 e2 79 18 bf 00 0a 	vbroadcastss 0xa00(%rdi),%xmm7
  4014b9:	00 00 
  4014bb:	c4 62 51 98 d0       	vfmadd132ps %xmm0,%xmm5,%xmm10
  4014c0:	c4 62 79 b8 d3       	vfmadd231ps %xmm3,%xmm0,%xmm10
  4014c5:	c4 62 79 18 89 00 0a 	vbroadcastss 0xa00(%rcx),%xmm9
  4014cc:	00 00 
  4014ce:	c4 42 79 18 81 00 0a 	vbroadcastss 0xa00(%r9),%xmm8
  4014d5:	00 00 
  4014d7:	c5 f8 28 b4 24 20 01 	vmovaps 0x120(%rsp),%xmm6
  4014de:	00 00 
  4014e0:	c5 f8 28 9c 24 10 01 	vmovaps 0x110(%rsp),%xmm3
  4014e7:	00 00 
  4014e9:	c4 62 69 b8 fe       	vfmadd231ps %xmm6,%xmm2,%xmm15
  4014ee:	c4 62 71 b8 f6       	vfmadd231ps %xmm6,%xmm1,%xmm14
  4014f3:	c4 62 79 b8 ee       	vfmadd231ps %xmm6,%xmm0,%xmm13
  4014f8:	c4 c2 79 18 a1 04 0a 	vbroadcastss 0xa04(%r9),%xmm4
  4014ff:	00 00 
  401501:	c4 62 31 b8 e3       	vfmadd231ps %xmm3,%xmm9,%xmm12
  401506:	c4 62 39 b8 db       	vfmadd231ps %xmm3,%xmm8,%xmm11
  40150b:	c4 62 41 b8 d3       	vfmadd231ps %xmm3,%xmm7,%xmm10
  401510:	c5 f8 28 b4 24 e0 00 	vmovaps 0xe0(%rsp),%xmm6
  401517:	00 00 
  401519:	c4 62 59 b8 5c 24 30 	vfmadd231ps 0x30(%rsp),%xmm4,%xmm11
  401520:	c4 e2 79 18 9f 04 0a 	vbroadcastss 0xa04(%rdi),%xmm3
  401527:	00 00 
  401529:	c4 62 31 b8 fe       	vfmadd231ps %xmm6,%xmm9,%xmm15
  40152e:	c4 62 39 b8 f6       	vfmadd231ps %xmm6,%xmm8,%xmm14
  401533:	c4 62 41 b8 ee       	vfmadd231ps %xmm6,%xmm7,%xmm13
  401538:	c4 62 59 b8 74 24 20 	vfmadd231ps 0x20(%rsp),%xmm4,%xmm14
  40153f:	c4 e2 79 18 b1 04 0a 	vbroadcastss 0xa04(%rcx),%xmm6
  401546:	00 00 
  401548:	c4 62 61 b8 6c 24 20 	vfmadd231ps 0x20(%rsp),%xmm3,%xmm13
  40154f:	c4 62 61 b8 54 24 30 	vfmadd231ps 0x30(%rsp),%xmm3,%xmm10
  401556:	c5 78 11 32          	vmovups %xmm14,(%rdx)
  40155a:	c4 62 49 b8 7c 24 20 	vfmadd231ps 0x20(%rsp),%xmm6,%xmm15
  401561:	c4 62 49 b8 64 24 30 	vfmadd231ps 0x30(%rsp),%xmm6,%xmm12
  401568:	c4 41 78 11 2a       	vmovups %xmm13,(%r10)
  40156d:	c5 78 11 38          	vmovups %xmm15,(%rax)
  401571:	c5 78 11 a0 00 14 00 	vmovups %xmm12,0x1400(%rax)
  401578:	00 
  401579:	c5 78 11 9a 00 14 00 	vmovups %xmm11,0x1400(%rdx)
  401580:	00 
  401581:	c4 41 78 11 92 00 14 	vmovups %xmm10,0x1400(%r10)
  401588:	00 00 
  40158a:	c5 78 28 6c 24 60    	vmovaps 0x60(%rsp),%xmm13
  401590:	c5 78 28 74 24 70    	vmovaps 0x70(%rsp),%xmm14
  401596:	c4 41 78 28 e5       	vmovaps %xmm13,%xmm12
  40159b:	c4 62 51 98 6c 24 40 	vfmadd132ps 0x40(%rsp),%xmm5,%xmm13
  4015a2:	c5 78 28 14 24       	vmovaps (%rsp),%xmm10
  4015a7:	c4 41 78 28 de       	vmovaps %xmm14,%xmm11
  4015ac:	c4 62 51 98 74 24 40 	vfmadd132ps 0x40(%rsp),%xmm5,%xmm14
  4015b3:	c5 78 28 bc 24 80 00 	vmovaps 0x80(%rsp),%xmm15
  4015ba:	00 00 
  4015bc:	c4 42 51 98 e2       	vfmadd132ps %xmm10,%xmm5,%xmm12
  4015c1:	c4 42 51 98 da       	vfmadd132ps %xmm10,%xmm5,%xmm11
  4015c6:	c4 62 69 b8 64 24 10 	vfmadd231ps 0x10(%rsp),%xmm2,%xmm12
  4015cd:	c4 e2 11 98 54 24 50 	vfmadd132ps 0x50(%rsp),%xmm13,%xmm2
  4015d4:	c4 62 71 b8 5c 24 10 	vfmadd231ps 0x10(%rsp),%xmm1,%xmm11
  4015db:	c5 78 28 6c 24 50    	vmovaps 0x50(%rsp),%xmm13
  4015e1:	c4 41 78 28 d7       	vmovaps %xmm15,%xmm10
  4015e6:	c4 62 51 98 7c 24 40 	vfmadd132ps 0x40(%rsp),%xmm5,%xmm15
  4015ed:	c4 c2 09 98 cd       	vfmadd132ps %xmm13,%xmm14,%xmm1
  4015f2:	c4 62 51 98 14 24    	vfmadd132ps (%rsp),%xmm5,%xmm10
  4015f8:	c4 62 79 b8 54 24 10 	vfmadd231ps 0x10(%rsp),%xmm0,%xmm10
  4015ff:	c4 c2 01 98 c5       	vfmadd132ps %xmm13,%xmm15,%xmm0
  401604:	c5 78 28 ac 24 a0 00 	vmovaps 0xa0(%rsp),%xmm13
  40160b:	00 00 
  40160d:	c4 42 31 b8 e5       	vfmadd231ps %xmm13,%xmm9,%xmm12
  401612:	c4 42 39 b8 dd       	vfmadd231ps %xmm13,%xmm8,%xmm11
  401617:	c4 42 41 b8 d5       	vfmadd231ps %xmm13,%xmm7,%xmm10
  40161c:	c5 78 28 ac 24 c0 00 	vmovaps 0xc0(%rsp),%xmm13
  401623:	00 00 
  401625:	c4 c2 41 b8 c5       	vfmadd231ps %xmm13,%xmm7,%xmm0
  40162a:	c5 f8 28 bc 24 00 01 	vmovaps 0x100(%rsp),%xmm7
  401631:	00 00 
  401633:	c4 c2 31 b8 d5       	vfmadd231ps %xmm13,%xmm9,%xmm2
  401638:	c4 c2 39 b8 cd       	vfmadd231ps %xmm13,%xmm8,%xmm1
  40163d:	c4 62 49 b8 e7       	vfmadd231ps %xmm7,%xmm6,%xmm12
  401642:	c4 62 59 b8 df       	vfmadd231ps %xmm7,%xmm4,%xmm11
  401647:	c4 62 61 b8 d7       	vfmadd231ps %xmm7,%xmm3,%xmm10
  40164c:	c5 f8 28 bc 24 90 00 	vmovaps 0x90(%rsp),%xmm7
  401653:	00 00 
  401655:	c5 78 11 a0 00 28 00 	vmovups %xmm12,0x2800(%rax)
  40165c:	00 
  40165d:	c4 e2 49 b8 d7       	vfmadd231ps %xmm7,%xmm6,%xmm2
  401662:	c4 e2 59 b8 cf       	vfmadd231ps %xmm7,%xmm4,%xmm1
  401667:	c4 e2 61 b8 c7       	vfmadd231ps %xmm7,%xmm3,%xmm0
  40166c:	c5 78 11 9a 00 28 00 	vmovups %xmm11,0x2800(%rdx)
  401673:	00 
  401674:	c4 41 78 11 92 00 28 	vmovups %xmm10,0x2800(%r10)
  40167b:	00 00 
  40167d:	c5 f8 11 8a 00 3c 00 	vmovups %xmm1,0x3c00(%rdx)
  401684:	00 
  401685:	c5 f8 11 90 00 3c 00 	vmovups %xmm2,0x3c00(%rax)
  40168c:	00 
  40168d:	c4 c1 78 11 82 00 3c 	vmovups %xmm0,0x3c00(%r10)
  401694:	00 00 
  401696:	0f 31                	rdtsc  
  401698:	41 83 c0 02          	add    $0x2,%r8d
  40169c:	41 81 f8 00 58 02 00 	cmp    $0x25800,%r8d
  4016a3:	0f 85 17 fd ff ff    	jne    4013c0 <main+0x1b0>
  4016a9:	48 89 da             	mov    %rbx,%rdx
  4016ac:	48 89 ee             	mov    %rbp,%rsi
  4016af:	4c 89 e7             	mov    %r12,%rdi
  4016b2:	4c 89 1c 24          	mov    %r11,(%rsp)
  4016b6:	e8 85 07 00 00       	callq  401e40 <_Z11encodeImagePfS_S_>
  4016bb:	4c 8b 1c 24          	mov    (%rsp),%r11
  4016bf:	4c 89 df             	mov    %r11,%rdi
  4016c2:	e8 c9 f9 ff ff       	callq  401090 <free@plt>
  4016c7:	4c 89 e7             	mov    %r12,%rdi
  4016ca:	e8 c1 f9 ff ff       	callq  401090 <free@plt>
  4016cf:	48 89 ef             	mov    %rbp,%rdi
  4016d2:	e8 b9 f9 ff ff       	callq  401090 <free@plt>
  4016d7:	48 89 df             	mov    %rbx,%rdi
  4016da:	e8 b1 f9 ff ff       	callq  401090 <free@plt>
  4016df:	48 81 c4 38 01 00 00 	add    $0x138,%rsp
  4016e6:	31 c0                	xor    %eax,%eax
  4016e8:	5b                   	pop    %rbx
  4016e9:	5d                   	pop    %rbp
  4016ea:	41 5c                	pop    %r12
  4016ec:	41 5d                	pop    %r13
  4016ee:	41 5e                	pop    %r14
  4016f0:	41 5f                	pop    %r15
  4016f2:	c3                   	retq   
  4016f3:	0f 1f 00             	nopl   (%rax)
  4016f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4016fd:	00 00 00 

0000000000401700 <_GLOBAL__sub_I__Z11decodeImagePfS_S_>:
  401700:	48 83 ec 08          	sub    $0x8,%rsp
  401704:	bf 41 51 60 00       	mov    $0x605141,%edi
  401709:	e8 c2 fa ff ff       	callq  4011d0 <_ZNSt8ios_base4InitC1Ev@plt>
  40170e:	ba 88 41 40 00       	mov    $0x404188,%edx
  401713:	be 41 51 60 00       	mov    $0x605141,%esi
  401718:	bf 10 11 40 00       	mov    $0x401110,%edi
  40171d:	48 83 c4 08          	add    $0x8,%rsp
  401721:	e9 5a fa ff ff       	jmpq   401180 <__cxa_atexit@plt>

0000000000401726 <_start>:
  401726:	31 ed                	xor    %ebp,%ebp
  401728:	49 89 d1             	mov    %rdx,%r9
  40172b:	5e                   	pop    %rsi
  40172c:	48 89 e2             	mov    %rsp,%rdx
  40172f:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401733:	50                   	push   %rax
  401734:	54                   	push   %rsp
  401735:	49 c7 c0 70 41 40 00 	mov    $0x404170,%r8
  40173c:	48 c7 c1 00 41 40 00 	mov    $0x404100,%rcx
  401743:	48 c7 c7 10 12 40 00 	mov    $0x401210,%rdi
  40174a:	e8 61 fa ff ff       	callq  4011b0 <__libc_start_main@plt>
  40174f:	f4                   	hlt    

0000000000401750 <deregister_tm_clones>:
  401750:	b8 0f 51 60 00       	mov    $0x60510f,%eax
  401755:	55                   	push   %rbp
  401756:	48 2d 08 51 60 00    	sub    $0x605108,%rax
  40175c:	48 83 f8 0e          	cmp    $0xe,%rax
  401760:	48 89 e5             	mov    %rsp,%rbp
  401763:	77 02                	ja     401767 <deregister_tm_clones+0x17>
  401765:	5d                   	pop    %rbp
  401766:	c3                   	retq   
  401767:	b8 00 00 00 00       	mov    $0x0,%eax
  40176c:	48 85 c0             	test   %rax,%rax
  40176f:	74 f4                	je     401765 <deregister_tm_clones+0x15>
  401771:	5d                   	pop    %rbp
  401772:	bf 08 51 60 00       	mov    $0x605108,%edi
  401777:	ff e0                	jmpq   *%rax
  401779:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401780 <register_tm_clones>:
  401780:	b8 08 51 60 00       	mov    $0x605108,%eax
  401785:	55                   	push   %rbp
  401786:	48 2d 08 51 60 00    	sub    $0x605108,%rax
  40178c:	48 c1 f8 03          	sar    $0x3,%rax
  401790:	48 89 e5             	mov    %rsp,%rbp
  401793:	48 89 c2             	mov    %rax,%rdx
  401796:	48 c1 ea 3f          	shr    $0x3f,%rdx
  40179a:	48 01 d0             	add    %rdx,%rax
  40179d:	48 d1 f8             	sar    %rax
  4017a0:	75 02                	jne    4017a4 <register_tm_clones+0x24>
  4017a2:	5d                   	pop    %rbp
  4017a3:	c3                   	retq   
  4017a4:	ba 00 00 00 00       	mov    $0x0,%edx
  4017a9:	48 85 d2             	test   %rdx,%rdx
  4017ac:	74 f4                	je     4017a2 <register_tm_clones+0x22>
  4017ae:	5d                   	pop    %rbp
  4017af:	48 89 c6             	mov    %rax,%rsi
  4017b2:	bf 08 51 60 00       	mov    $0x605108,%edi
  4017b7:	ff e2                	jmpq   *%rdx
  4017b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004017c0 <__do_global_dtors_aux>:
  4017c0:	80 3d 79 39 20 00 00 	cmpb   $0x0,0x203979(%rip)        # 605140 <completed.6355>
  4017c7:	75 11                	jne    4017da <__do_global_dtors_aux+0x1a>
  4017c9:	55                   	push   %rbp
  4017ca:	48 89 e5             	mov    %rsp,%rbp
  4017cd:	e8 7e ff ff ff       	callq  401750 <deregister_tm_clones>
  4017d2:	5d                   	pop    %rbp
  4017d3:	c6 05 66 39 20 00 01 	movb   $0x1,0x203966(%rip)        # 605140 <completed.6355>
  4017da:	f3 c3                	repz retq 
  4017dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004017e0 <frame_dummy>:
  4017e0:	48 83 3d 08 35 20 00 	cmpq   $0x0,0x203508(%rip)        # 604cf0 <__JCR_END__>
  4017e7:	00 
  4017e8:	74 1e                	je     401808 <frame_dummy+0x28>
  4017ea:	b8 00 00 00 00       	mov    $0x0,%eax
  4017ef:	48 85 c0             	test   %rax,%rax
  4017f2:	74 14                	je     401808 <frame_dummy+0x28>
  4017f4:	55                   	push   %rbp
  4017f5:	bf f0 4c 60 00       	mov    $0x604cf0,%edi
  4017fa:	48 89 e5             	mov    %rsp,%rbp
  4017fd:	ff d0                	callq  *%rax
  4017ff:	5d                   	pop    %rbp
  401800:	e9 7b ff ff ff       	jmpq   401780 <register_tm_clones>
  401805:	0f 1f 00             	nopl   (%rax)
  401808:	e9 73 ff ff ff       	jmpq   401780 <register_tm_clones>
  40180d:	0f 1f 00             	nopl   (%rax)

0000000000401810 <_Z20generateCoefficientsPf>:
  401810:	55                   	push   %rbp
  401811:	48 b8 00 00 00 00 ff 	movabs $0xffffffff00000000,%rax
  401818:	ff ff ff 
  40181b:	48 be 00 00 00 00 00 	movabs $0x3f80000000000000,%rsi
  401822:	00 80 3f 
  401825:	49 b8 00 00 00 00 ab 	movabs $0x3f2aaaab00000000,%r8
  40182c:	aa 2a 3f 
  40182f:	48 b9 00 00 00 00 ab 	movabs $0x3eaaaaab00000000,%rcx
  401836:	aa aa 3e 
  401839:	49 b9 00 00 00 00 39 	movabs $0x3e638e3900000000,%r9
  401840:	8e 63 3e 
  401843:	48 89 e5             	mov    %rsp,%rbp
  401846:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40184a:	49 ba 00 00 00 00 39 	movabs $0x3de38e3900000000,%r10
  401851:	8e e3 3d 
  401854:	48 83 ec 48          	sub    $0x48,%rsp
  401858:	49 bb 00 00 00 00 39 	movabs $0x3ee38e3900000000,%r11
  40185f:	8e e3 3e 
  401862:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  401869:	00 00 
  40186b:	48 8b 54 24 c0       	mov    -0x40(%rsp),%rdx
  401870:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  401877:	00 00 
  401879:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  401880:	00 00 
  401882:	c5 f8 28 05 66 29 00 	vmovaps 0x2966(%rip),%xmm0        # 4041f0 <__dso_handle+0x68>
  401889:	00 
  40188a:	48 21 c2             	and    %rax,%rdx
  40188d:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  401894:	00 00 
  401896:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  40189d:	00 00 
  40189f:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  4018a6:	48 89 4c 24 b8       	mov    %rcx,-0x48(%rsp)
  4018ab:	4c 89 4c 24 a8       	mov    %r9,-0x58(%rsp)
  4018b0:	89 d2                	mov    %edx,%edx
  4018b2:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  4018b9:	00 00 
  4018bb:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  4018c2:	00 00 
  4018c4:	48 09 f2             	or     %rsi,%rdx
  4018c7:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  4018ce:	00 00 
  4018d0:	4c 89 54 24 98       	mov    %r10,-0x68(%rsp)
  4018d5:	48 89 54 24 c0       	mov    %rdx,-0x40(%rsp)
  4018da:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  4018df:	c5 f8 29 07          	vmovaps %xmm0,(%rdi)
  4018e3:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  4018ea:	00 00 
  4018ec:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  4018f3:	00 00 
  4018f5:	48 21 c2             	and    %rax,%rdx
  4018f8:	48 81 ca 39 8e e3 3e 	or     $0x3ee38e39,%rdx
  4018ff:	89 d2                	mov    %edx,%edx
  401901:	c5 f8 28 05 f7 28 00 	vmovaps 0x28f7(%rip),%xmm0        # 404200 <__dso_handle+0x78>
  401908:	00 
  401909:	4c 09 c2             	or     %r8,%rdx
  40190c:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  401911:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  401916:	c5 f8 29 47 10       	vmovaps %xmm0,0x10(%rdi)
  40191b:	48 21 c2             	and    %rax,%rdx
  40191e:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  401925:	c5 f8 28 05 e3 28 00 	vmovaps 0x28e3(%rip),%xmm0        # 404210 <__dso_handle+0x88>
  40192c:	00 
  40192d:	89 d2                	mov    %edx,%edx
  40192f:	48 09 ca             	or     %rcx,%rdx
  401932:	c5 f8 29 47 20       	vmovaps %xmm0,0x20(%rdi)
  401937:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  40193b:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  401940:	c5 f8 28 4c 24 b8    	vmovaps -0x48(%rsp),%xmm1
  401946:	c5 f8 28 54 24 a8    	vmovaps -0x58(%rsp),%xmm2
  40194c:	c5 f8 29 4f 40       	vmovaps %xmm1,0x40(%rdi)
  401951:	c5 f8 29 47 30       	vmovaps %xmm0,0x30(%rdi)
  401956:	c5 f8 29 57 50       	vmovaps %xmm2,0x50(%rdi)
  40195b:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  401962:	00 00 
  401964:	48 8b 54 24 a8       	mov    -0x58(%rsp),%rdx
  401969:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  401970:	00 00 
  401972:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  401979:	00 00 
  40197b:	c5 f8 28 5c 24 98    	vmovaps -0x68(%rsp),%xmm3
  401981:	48 21 c2             	and    %rax,%rdx
  401984:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  40198b:	00 00 
  40198d:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  401994:	00 00 
  401996:	48 81 ca ab aa aa 3e 	or     $0x3eaaaaab,%rdx
  40199d:	c5 f8 28 64 24 88    	vmovaps -0x78(%rsp),%xmm4
  4019a3:	89 d2                	mov    %edx,%edx
  4019a5:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  4019ac:	00 00 
  4019ae:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  4019b5:	00 00 
  4019b7:	4c 09 ca             	or     %r9,%rdx
  4019ba:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  4019c1:	00 00 
  4019c3:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  4019ca:	00 00 
  4019cc:	48 89 54 24 a8       	mov    %rdx,-0x58(%rsp)
  4019d1:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  4019d6:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  4019dd:	00 00 
  4019df:	c5 f8 28 6c 24 b8    	vmovaps -0x48(%rsp),%xmm5
  4019e5:	48 21 c2             	and    %rax,%rdx
  4019e8:	48 81 ca 39 8e e3 3d 	or     $0x3de38e39,%rdx
  4019ef:	81 e2 39 8e e3 3d    	and    $0x3de38e39,%edx
  4019f5:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  4019fa:	48 8b 54 24 98       	mov    -0x68(%rsp),%rdx
  4019ff:	c5 f8 28 74 24 a8    	vmovaps -0x58(%rsp),%xmm6
  401a05:	48 21 c2             	and    %rax,%rdx
  401a08:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  401a0f:	89 d2                	mov    %edx,%edx
  401a11:	4c 09 da             	or     %r11,%rdx
  401a14:	48 89 54 24 98       	mov    %rdx,-0x68(%rsp)
  401a19:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  401a1e:	c5 f8 29 5f 60       	vmovaps %xmm3,0x60(%rdi)
  401a23:	48 21 c2             	and    %rax,%rdx
  401a26:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  401a2d:	81 e2 39 8e 63 3e    	and    $0x3e638e39,%edx
  401a33:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  401a38:	48 8b 54 24 88       	mov    -0x78(%rsp),%rdx
  401a3d:	c5 f8 28 7c 24 98    	vmovaps -0x68(%rsp),%xmm7
  401a43:	48 21 c2             	and    %rax,%rdx
  401a46:	48 81 ca 00 00 80 3f 	or     $0x3f800000,%rdx
  401a4d:	89 d2                	mov    %edx,%edx
  401a4f:	4c 09 c2             	or     %r8,%rdx
  401a52:	48 89 54 24 88       	mov    %rdx,-0x78(%rsp)
  401a57:	48 8b 54 24 90       	mov    -0x70(%rsp),%rdx
  401a5c:	c5 f8 29 67 70       	vmovaps %xmm4,0x70(%rdi)
  401a61:	48 21 c2             	and    %rax,%rdx
  401a64:	48 81 ca ab aa aa 3e 	or     $0x3eaaaaab,%rdx
  401a6b:	81 e2 ab aa aa 3e    	and    $0x3eaaaaab,%edx
  401a71:	48 89 54 24 90       	mov    %rdx,-0x70(%rsp)
  401a76:	c5 f8 28 4c 24 88    	vmovaps -0x78(%rsp),%xmm1
  401a7c:	c5 f8 29 af 80 00 00 	vmovaps %xmm5,0x80(%rdi)
  401a83:	00 
  401a84:	c5 f8 29 b7 90 00 00 	vmovaps %xmm6,0x90(%rdi)
  401a8b:	00 
  401a8c:	c5 f8 29 bf a0 00 00 	vmovaps %xmm7,0xa0(%rdi)
  401a93:	00 
  401a94:	c5 f8 29 8f b0 00 00 	vmovaps %xmm1,0xb0(%rdi)
  401a9b:	00 
  401a9c:	48 c7 44 24 b0 00 00 	movq   $0x0,-0x50(%rsp)
  401aa3:	00 00 
  401aa5:	48 8b 54 24 b0       	mov    -0x50(%rsp),%rdx
  401aaa:	48 c7 44 24 a0 00 00 	movq   $0x0,-0x60(%rsp)
  401ab1:	00 00 
  401ab3:	48 c7 44 24 90 00 00 	movq   $0x0,-0x70(%rsp)
  401aba:	00 00 
  401abc:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  401ac3:	00 00 
  401ac5:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  401acc:	00 00 
  401ace:	48 21 c2             	and    %rax,%rdx
  401ad1:	48 c7 44 24 a8 00 00 	movq   $0x0,-0x58(%rsp)
  401ad8:	00 00 
  401ada:	48 c7 44 24 98 00 00 	movq   $0x0,-0x68(%rsp)
  401ae1:	00 00 
  401ae3:	48 81 ca 39 8e 63 3e 	or     $0x3e638e39,%rdx
  401aea:	48 c7 44 24 88 00 00 	movq   $0x0,-0x78(%rsp)
  401af1:	00 00 
  401af3:	48 c7 44 24 b8 00 00 	movq   $0x0,-0x48(%rsp)
  401afa:	00 00 
  401afc:	89 d2                	mov    %edx,%edx
  401afe:	48 c7 44 24 c0 00 00 	movq   $0x0,-0x40(%rsp)
  401b05:	00 00 
  401b07:	4c 89 54 24 a8       	mov    %r10,-0x58(%rsp)
  401b0c:	48 09 ca             	or     %rcx,%rdx
  401b0f:	4c 89 4c 24 98       	mov    %r9,-0x68(%rsp)
  401b14:	48 89 4c 24 88       	mov    %rcx,-0x78(%rsp)
  401b19:	48 89 54 24 b0       	mov    %rdx,-0x50(%rsp)
  401b1e:	48 8b 54 24 a0       	mov    -0x60(%rsp),%rdx
  401b23:	c5 f8 28 54 24 b8    	vmovaps -0x48(%rsp),%xmm2
  401b29:	48 21 c2             	and    %rax,%rdx
  401b2c:	c5 f8 28 5c 24 a8    	vmovaps -0x58(%rsp),%xmm3
  401b32:	48 81 ca 39 8e e3 3e 	or     $0x3ee38e39,%rdx
  401b39:	89 d2                	mov    %edx,%edx
  401b3b:	4c 09 c2             	or     %r8,%rdx
  401b3e:	48 89 54 24 a0       	mov    %rdx,-0x60(%rsp)
  401b43:	48 8b 54 24 90       	mov    -0x70(%rsp),%rdx
  401b48:	c5 f8 28 64 24 98    	vmovaps -0x68(%rsp),%xmm4
  401b4e:	48 21 c2             	and    %rax,%rdx
  401b51:	48 81 ca ab aa 2a 3f 	or     $0x3f2aaaab,%rdx
  401b58:	89 d2                	mov    %edx,%edx
  401b5a:	48 09 f2             	or     %rsi,%rdx
  401b5d:	48 89 54 24 90       	mov    %rdx,-0x70(%rsp)
  401b62:	c5 f8 28 6c 24 88    	vmovaps -0x78(%rsp),%xmm5
  401b68:	c5 f8 29 97 c0 00 00 	vmovaps %xmm2,0xc0(%rdi)
  401b6f:	00 
  401b70:	c5 f8 29 9f d0 00 00 	vmovaps %xmm3,0xd0(%rdi)
  401b77:	00 
  401b78:	c5 f8 29 a7 e0 00 00 	vmovaps %xmm4,0xe0(%rdi)
  401b7f:	00 
  401b80:	c5 f8 29 af f0 00 00 	vmovaps %xmm5,0xf0(%rdi)
  401b87:	00 
  401b88:	c9                   	leaveq 
  401b89:	c3                   	retq   
  401b8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401b90 <_Z6kernelPfS_S_S_S_S_S_i>:
  401b90:	41 57                	push   %r15
  401b92:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  401b96:	c4 62 79 18 0f       	vbroadcastss (%rdi),%xmm9
  401b9b:	41 56                	push   %r14
  401b9d:	c4 62 79 18 06       	vbroadcastss (%rsi),%xmm8
  401ba2:	41 55                	push   %r13
  401ba4:	c4 e2 79 18 12       	vbroadcastss (%rdx),%xmm2
  401ba9:	41 54                	push   %r12
  401bab:	55                   	push   %rbp
  401bac:	53                   	push   %rbx
  401bad:	4c 63 5c 24 40       	movslq 0x40(%rsp),%r11
  401bb2:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  401bb7:	c4 e2 79 18 77 04    	vbroadcastss 0x4(%rdi),%xmm6
  401bbd:	44 89 db             	mov    %r11d,%ebx
  401bc0:	c5 f8 28 08          	vmovaps (%rax),%xmm1
  401bc4:	c1 eb 1f             	shr    $0x1f,%ebx
  401bc7:	c5 78 28 e6          	vmovaps %xmm6,%xmm12
  401bcb:	c4 e2 79 18 6e 04    	vbroadcastss 0x4(%rsi),%xmm5
  401bd1:	44 01 db             	add    %r11d,%ebx
  401bd4:	c4 e2 79 98 d1       	vfmadd132ps %xmm1,%xmm0,%xmm2
  401bd9:	d1 fb                	sar    %ebx
  401bdb:	c4 62 79 98 c9       	vfmadd132ps %xmm1,%xmm0,%xmm9
  401be0:	c4 62 79 98 c1       	vfmadd132ps %xmm1,%xmm0,%xmm8
  401be5:	c5 f8 28 60 10       	vmovaps 0x10(%rax),%xmm4
  401bea:	c5 78 28 dd          	vmovaps %xmm5,%xmm11
  401bee:	48 63 db             	movslq %ebx,%rbx
  401bf1:	4e 8d 14 9d 00 00 00 	lea    0x0(,%r11,4),%r10
  401bf8:	00 
  401bf9:	c4 e2 79 18 4a 04    	vbroadcastss 0x4(%rdx),%xmm1
  401bff:	48 c1 e3 02          	shl    $0x2,%rbx
  401c03:	c4 62 79 98 e4       	vfmadd132ps %xmm4,%xmm0,%xmm12
  401c08:	c4 62 79 98 dc       	vfmadd132ps %xmm4,%xmm0,%xmm11
  401c0d:	4c 8d 34 1f          	lea    (%rdi,%rbx,1),%r14
  401c11:	4c 8d 2c 1e          	lea    (%rsi,%rbx,1),%r13
  401c15:	c5 f8 28 78 40       	vmovaps 0x40(%rax),%xmm7
  401c1a:	c4 e2 79 98 e1       	vfmadd132ps %xmm1,%xmm0,%xmm4
  401c1f:	4c 8d 24 1a          	lea    (%rdx,%rbx,1),%r12
  401c23:	c5 78 28 d4          	vmovaps %xmm4,%xmm10
  401c27:	48 83 c3 04          	add    $0x4,%rbx
  401c2b:	c5 f8 28 60 50       	vmovaps 0x50(%rax),%xmm4
  401c30:	4c 8d 3c 1f          	lea    (%rdi,%rbx,1),%r15
  401c34:	48 8d 2c 1e          	lea    (%rsi,%rbx,1),%rbp
  401c38:	c4 62 49 b8 cf       	vfmadd231ps %xmm7,%xmm6,%xmm9
  401c3d:	c4 62 51 b8 c7       	vfmadd231ps %xmm7,%xmm5,%xmm8
  401c42:	c4 e2 69 98 f9       	vfmadd132ps %xmm1,%xmm2,%xmm7
  401c47:	c5 f8 28 98 90 00 00 	vmovaps 0x90(%rax),%xmm3
  401c4e:	00 
  401c4f:	c4 e2 19 98 f4       	vfmadd132ps %xmm4,%xmm12,%xmm6
  401c54:	c4 e2 21 98 ec       	vfmadd132ps %xmm4,%xmm11,%xmm5
  401c59:	c4 e2 29 98 e1       	vfmadd132ps %xmm1,%xmm10,%xmm4
  401c5e:	48 01 d3             	add    %rdx,%rbx
  401c61:	c4 42 79 18 26       	vbroadcastss (%r14),%xmm12
  401c66:	c4 42 79 18 5d 00    	vbroadcastss 0x0(%r13),%xmm11
  401c6c:	c4 e2 19 b8 f3       	vfmadd231ps %xmm3,%xmm12,%xmm6
  401c71:	c4 42 79 18 14 24    	vbroadcastss (%r12),%xmm10
  401c77:	c4 e2 21 b8 eb       	vfmadd231ps %xmm3,%xmm11,%xmm5
  401c7c:	c5 f8 28 90 80 00 00 	vmovaps 0x80(%rax),%xmm2
  401c83:	00 
  401c84:	c4 e2 29 b8 e3       	vfmadd231ps %xmm3,%xmm10,%xmm4
  401c89:	c4 e2 29 b8 fa       	vfmadd231ps %xmm2,%xmm10,%xmm7
  401c8e:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  401c93:	c4 62 79 18 13       	vbroadcastss (%rbx),%xmm10
  401c98:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  401c9d:	c4 42 79 18 27       	vbroadcastss (%r15),%xmm12
  401ca2:	c4 62 79 18 5d 00    	vbroadcastss 0x0(%rbp),%xmm11
  401ca8:	c5 f8 28 90 c0 00 00 	vmovaps 0xc0(%rax),%xmm2
  401caf:	00 
  401cb0:	c5 f8 28 88 d0 00 00 	vmovaps 0xd0(%rax),%xmm1
  401cb7:	00 
  401cb8:	c4 62 19 b8 ca       	vfmadd231ps %xmm2,%xmm12,%xmm9
  401cbd:	c4 62 21 b8 c2       	vfmadd231ps %xmm2,%xmm11,%xmm8
  401cc2:	c4 c2 41 98 d2       	vfmadd132ps %xmm10,%xmm7,%xmm2
  401cc7:	c5 78 29 09          	vmovaps %xmm9,(%rcx)
  401ccb:	c4 62 49 98 e1       	vfmadd132ps %xmm1,%xmm6,%xmm12
  401cd0:	c4 62 51 98 d9       	vfmadd132ps %xmm1,%xmm5,%xmm11
  401cd5:	4c 01 d1             	add    %r10,%rcx
  401cd8:	c4 62 59 98 d1       	vfmadd132ps %xmm1,%xmm4,%xmm10
  401cdd:	c4 41 78 29 00       	vmovaps %xmm8,(%r8)
  401ce2:	4d 01 d0             	add    %r10,%r8
  401ce5:	c4 c1 78 29 11       	vmovaps %xmm2,(%r9)
  401cea:	4d 01 d1             	add    %r10,%r9
  401ced:	c5 78 29 21          	vmovaps %xmm12,(%rcx)
  401cf1:	4c 01 d1             	add    %r10,%rcx
  401cf4:	c4 41 78 29 18       	vmovaps %xmm11,(%r8)
  401cf9:	4d 01 d0             	add    %r10,%r8
  401cfc:	4d 01 ca             	add    %r9,%r10
  401cff:	c4 41 78 29 11       	vmovaps %xmm10,(%r9)
  401d04:	c4 e2 79 18 3f       	vbroadcastss (%rdi),%xmm7
  401d09:	c4 e2 79 18 36       	vbroadcastss (%rsi),%xmm6
  401d0e:	c5 78 28 cf          	vmovaps %xmm7,%xmm9
  401d12:	c5 f8 28 48 30       	vmovaps 0x30(%rax),%xmm1
  401d17:	c5 78 28 c6          	vmovaps %xmm6,%xmm8
  401d1b:	c4 e2 79 18 1a       	vbroadcastss (%rdx),%xmm3
  401d20:	c4 e2 79 98 f9       	vfmadd132ps %xmm1,%xmm0,%xmm7
  401d25:	c4 e2 79 98 f1       	vfmadd132ps %xmm1,%xmm0,%xmm6
  401d2a:	c5 78 28 df          	vmovaps %xmm7,%xmm11
  401d2e:	c5 78 28 d6          	vmovaps %xmm6,%xmm10
  401d32:	c5 f8 28 50 20       	vmovaps 0x20(%rax),%xmm2
  401d37:	c4 e2 79 18 7f 04    	vbroadcastss 0x4(%rdi),%xmm7
  401d3d:	c4 62 79 98 ca       	vfmadd132ps %xmm2,%xmm0,%xmm9
  401d42:	c4 62 79 98 c2       	vfmadd132ps %xmm2,%xmm0,%xmm8
  401d47:	c4 e2 79 98 d3       	vfmadd132ps %xmm3,%xmm0,%xmm2
  401d4c:	c4 e2 61 b8 c1       	vfmadd231ps %xmm1,%xmm3,%xmm0
  401d51:	c5 78 28 e2          	vmovaps %xmm2,%xmm12
  401d55:	c4 e2 79 18 76 04    	vbroadcastss 0x4(%rsi),%xmm6
  401d5b:	c5 f8 28 50 60       	vmovaps 0x60(%rax),%xmm2
  401d60:	c4 e2 79 18 5a 04    	vbroadcastss 0x4(%rdx),%xmm3
  401d66:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  401d6b:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  401d70:	c5 f8 28 60 70       	vmovaps 0x70(%rax),%xmm4
  401d75:	c4 62 61 b8 e2       	vfmadd231ps %xmm2,%xmm3,%xmm12
  401d7a:	c4 c2 79 18 2c 24    	vbroadcastss (%r12),%xmm5
  401d80:	c4 62 41 b8 dc       	vfmadd231ps %xmm4,%xmm7,%xmm11
  401d85:	c4 62 49 b8 d4       	vfmadd231ps %xmm4,%xmm6,%xmm10
  401d8a:	c4 e2 79 98 e3       	vfmadd132ps %xmm3,%xmm0,%xmm4
  401d8f:	c4 c2 79 18 3e       	vbroadcastss (%r14),%xmm7
  401d94:	c4 c2 79 18 75 00    	vbroadcastss 0x0(%r13),%xmm6
  401d9a:	c5 f8 28 90 a0 00 00 	vmovaps 0xa0(%rax),%xmm2
  401da1:	00 
  401da2:	c5 f8 28 98 b0 00 00 	vmovaps 0xb0(%rax),%xmm3
  401da9:	00 
  401daa:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  401daf:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  401db4:	c4 62 51 b8 e2       	vfmadd231ps %xmm2,%xmm5,%xmm12
  401db9:	c5 f8 28 88 f0 00 00 	vmovaps 0xf0(%rax),%xmm1
  401dc0:	00 
  401dc1:	c4 62 41 b8 db       	vfmadd231ps %xmm3,%xmm7,%xmm11
  401dc6:	c4 62 49 b8 d3       	vfmadd231ps %xmm3,%xmm6,%xmm10
  401dcb:	c4 e2 59 98 dd       	vfmadd132ps %xmm5,%xmm4,%xmm3
  401dd0:	c4 c2 79 18 3f       	vbroadcastss (%r15),%xmm7
  401dd5:	c4 e2 79 18 75 00    	vbroadcastss 0x0(%rbp),%xmm6
  401ddb:	c4 e2 79 18 2b       	vbroadcastss (%rbx),%xmm5
  401de0:	c5 f8 28 90 e0 00 00 	vmovaps 0xe0(%rax),%xmm2
  401de7:	00 
  401de8:	c4 62 41 b8 ca       	vfmadd231ps %xmm2,%xmm7,%xmm9
  401ded:	c4 62 49 b8 c2       	vfmadd231ps %xmm2,%xmm6,%xmm8
  401df2:	c5 78 29 09          	vmovaps %xmm9,(%rcx)
  401df6:	c4 e2 19 98 d5       	vfmadd132ps %xmm5,%xmm12,%xmm2
  401dfb:	c4 e2 21 98 f9       	vfmadd132ps %xmm1,%xmm11,%xmm7
  401e00:	c4 e2 29 98 f1       	vfmadd132ps %xmm1,%xmm10,%xmm6
  401e05:	c4 41 78 29 00       	vmovaps %xmm8,(%r8)
  401e0a:	c4 e2 61 98 e9       	vfmadd132ps %xmm1,%xmm3,%xmm5
  401e0f:	c4 c1 78 29 12       	vmovaps %xmm2,(%r10)
  401e14:	c4 81 78 29 34 98    	vmovaps %xmm6,(%r8,%r11,4)
  401e1a:	c4 a1 78 29 3c 99    	vmovaps %xmm7,(%rcx,%r11,4)
  401e20:	c4 81 78 29 2c 9a    	vmovaps %xmm5,(%r10,%r11,4)
  401e26:	5b                   	pop    %rbx
  401e27:	5d                   	pop    %rbp
  401e28:	41 5c                	pop    %r12
  401e2a:	41 5d                	pop    %r13
  401e2c:	41 5e                	pop    %r14
  401e2e:	41 5f                	pop    %r15
  401e30:	c3                   	retq   
  401e31:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401e36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401e3d:	00 00 00 

0000000000401e40 <_Z11encodeImagePfS_S_>:
  401e40:	55                   	push   %rbp
  401e41:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  401e45:	48 89 e5             	mov    %rsp,%rbp
  401e48:	41 54                	push   %r12
  401e4a:	53                   	push   %rbx
  401e4b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  401e4f:	48 81 ec 10 02 00 00 	sub    $0x210,%rsp
  401e56:	48 8d 84 24 98 00 00 	lea    0x98(%rsp),%rax
  401e5d:	00 
  401e5e:	c5 f9 6f 0d ca 23 00 	vmovdqa 0x23ca(%rip),%xmm1        # 404230 <__dso_handle+0xa8>
  401e65:	00 
  401e66:	48 89 bc 24 00 01 00 	mov    %rdi,0x100(%rsp)
  401e6d:	00 
  401e6e:	48 89 bc 24 10 01 00 	mov    %rdi,0x110(%rsp)
  401e75:	00 
  401e76:	48 81 c7 00 00 4b 00 	add    $0x4b0000,%rdi
  401e7d:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  401e84:	00 
  401e85:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  401e8c:	00 
  401e8d:	48 89 bc 24 20 01 00 	mov    %rdi,0x120(%rsp)
  401e94:	00 
  401e95:	c5 f9 7f 84 24 e0 00 	vmovdqa %xmm0,0xe0(%rsp)
  401e9c:	00 00 
  401e9e:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  401ea5:	00 
  401ea6:	48 8d 84 24 f8 00 00 	lea    0xf8(%rsp),%rax
  401ead:	00 
  401eae:	48 89 bc 24 18 01 00 	mov    %rdi,0x118(%rsp)
  401eb5:	00 
  401eb6:	c5 f9 7f 8c 24 f0 00 	vmovdqa %xmm1,0xf0(%rsp)
  401ebd:	00 00 
  401ebf:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  401ec4:	c5 f9 6f 05 54 23 00 	vmovdqa 0x2354(%rip),%xmm0        # 404220 <__dso_handle+0x98>
  401ecb:	00 
  401ecc:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
  401ed3:	00 
  401ed4:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  401edb:	00 
  401edc:	48 89 b4 24 60 01 00 	mov    %rsi,0x160(%rsp)
  401ee3:	00 
  401ee4:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  401eeb:	00 00 
  401eed:	48 89 84 24 38 01 00 	mov    %rax,0x138(%rsp)
  401ef4:	00 
  401ef5:	48 8d 84 24 58 01 00 	lea    0x158(%rsp),%rax
  401efc:	00 
  401efd:	48 89 b4 24 70 01 00 	mov    %rsi,0x170(%rsp)
  401f04:	00 
  401f05:	48 81 c6 00 00 4b 00 	add    $0x4b0000,%rsi
  401f0c:	c5 f9 6f 05 2c 23 00 	vmovdqa 0x232c(%rip),%xmm0        # 404240 <__dso_handle+0xb8>
  401f13:	00 
  401f14:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  401f1b:	00 00 
  401f1d:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
  401f24:	00 00 
  401f26:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  401f2d:	00 00 
  401f2f:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  401f36:	00 00 00 00 00 
  401f3b:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  401f42:	00 00 00 00 00 
  401f47:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  401f4e:	00 00 00 00 00 
  401f53:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  401f5a:	00 00 00 00 00 
  401f5f:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  401f66:	00 00 00 00 00 
  401f6b:	48 c7 84 24 c8 00 00 	movq   $0x0,0xc8(%rsp)
  401f72:	00 00 00 00 00 
  401f77:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  401f7e:	00 00 00 00 00 
  401f83:	48 c7 84 24 28 01 00 	movq   $0x0,0x128(%rsp)
  401f8a:	00 00 00 00 00 
  401f8f:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  401f96:	00 00 00 00 00 
  401f9b:	c5 f9 7f 84 24 40 01 	vmovdqa %xmm0,0x140(%rsp)
  401fa2:	00 00 
  401fa4:	48 c7 84 24 88 01 00 	movq   $0x0,0x188(%rsp)
  401fab:	00 00 00 00 00 
  401fb0:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
  401fb7:	00 
  401fb8:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  401fbf:	00 
  401fc0:	c5 f9 7f 8c 24 50 01 	vmovdqa %xmm1,0x150(%rsp)
  401fc7:	00 00 
  401fc9:	c5 f9 7f 84 24 a0 01 	vmovdqa %xmm0,0x1a0(%rsp)
  401fd0:	00 00 
  401fd2:	c5 f9 7f 8c 24 b0 01 	vmovdqa %xmm1,0x1b0(%rsp)
  401fd9:	00 00 
  401fdb:	c5 f9 7f 84 24 00 02 	vmovdqa %xmm0,0x200(%rsp)
  401fe2:	00 00 
  401fe4:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  401feb:	00 
  401fec:	48 8d 84 24 b8 01 00 	lea    0x1b8(%rsp),%rax
  401ff3:	00 
  401ff4:	48 89 94 24 c0 01 00 	mov    %rdx,0x1c0(%rsp)
  401ffb:	00 
  401ffc:	48 89 94 24 d0 01 00 	mov    %rdx,0x1d0(%rsp)
  402003:	00 
  402004:	48 81 c2 00 00 4b 00 	add    $0x4b0000,%rdx
  40200b:	48 89 b4 24 80 01 00 	mov    %rsi,0x180(%rsp)
  402012:	00 
  402013:	48 89 84 24 f0 01 00 	mov    %rax,0x1f0(%rsp)
  40201a:	00 
  40201b:	48 89 94 24 e0 01 00 	mov    %rdx,0x1e0(%rsp)
  402022:	00 
  402023:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  40202a:	00 
  40202b:	48 89 94 24 d8 01 00 	mov    %rdx,0x1d8(%rsp)
  402032:	00 
  402033:	48 8d 94 24 b0 01 00 	lea    0x1b0(%rsp),%rdx
  40203a:	00 
  40203b:	48 89 b4 24 78 01 00 	mov    %rsi,0x178(%rsp)
  402042:	00 
  402043:	31 f6                	xor    %esi,%esi
  402045:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  40204c:	00 00 00 00 00 
  402051:	48 c7 84 24 e8 01 00 	movq   $0x0,0x1e8(%rsp)
  402058:	00 00 00 00 00 
  40205d:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
  402064:	00 
  402065:	e8 d6 10 00 00       	callq  403140 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  40206a:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  40206f:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  402074:	48 39 c7             	cmp    %rax,%rdi
  402077:	0f 84 72 05 00 00    	je     4025ef <_Z11encodeImagePfS_S_+0x7af>
  40207d:	48 85 ff             	test   %rdi,%rdi
  402080:	0f 84 00 05 00 00    	je     402586 <_Z11encodeImagePfS_S_+0x746>
  402086:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  40208d:	00 00 
  40208f:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402093:	48 8b 94 24 70 01 00 	mov    0x170(%rsp),%rdx
  40209a:	00 
  40209b:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  4020a2:	00 
  4020a3:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4020a7:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  4020ab:	48 8b 94 24 78 01 00 	mov    0x178(%rsp),%rdx
  4020b2:	00 
  4020b3:	48 89 47 10          	mov    %rax,0x10(%rdi)
  4020b7:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  4020be:	00 
  4020bf:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  4020c4:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  4020c8:	48 8b 94 24 80 01 00 	mov    0x180(%rsp),%rdx
  4020cf:	00 
  4020d0:	48 85 c0             	test   %rax,%rax
  4020d3:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  4020d7:	48 8b 94 24 88 01 00 	mov    0x188(%rsp),%rdx
  4020de:	00 
  4020df:	48 89 47 18          	mov    %rax,0x18(%rdi)
  4020e3:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  4020e7:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  4020eb:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  4020ef:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  4020f3:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  4020f7:	74 04                	je     4020fd <_Z11encodeImagePfS_S_+0x2bd>
  4020f9:	f0 83 00 01          	lock addl $0x1,(%rax)
  4020fd:	83 bc 24 54 01 00 00 	cmpl   $0x2,0x154(%rsp)
  402104:	02 
  402105:	0f 8f 15 04 00 00    	jg     402520 <_Z11encodeImagePfS_S_+0x6e0>
  40210b:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  402112:	00 
  402113:	48 8b 47 48          	mov    0x48(%rdi),%rax
  402117:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40211c:	48 8b 12             	mov    (%rdx),%rdx
  40211f:	48 89 10             	mov    %rdx,(%rax)
  402122:	48 8b 94 24 98 01 00 	mov    0x198(%rsp),%rdx
  402129:	00 
  40212a:	48 8b 47 48          	mov    0x48(%rdi),%rax
  40212e:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402133:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402137:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40213b:	48 83 c7 60          	add    $0x60,%rdi
  40213f:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  402144:	48 39 fe             	cmp    %rdi,%rsi
  402147:	0f 84 8b 04 00 00    	je     4025d8 <_Z11encodeImagePfS_S_+0x798>
  40214d:	48 85 ff             	test   %rdi,%rdi
  402150:	0f 84 b0 00 00 00    	je     402206 <_Z11encodeImagePfS_S_+0x3c6>
  402156:	c5 f9 6f 84 24 f0 00 	vmovdqa 0xf0(%rsp),%xmm0
  40215d:	00 00 
  40215f:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  402163:	48 8b 94 24 10 01 00 	mov    0x110(%rsp),%rdx
  40216a:	00 
  40216b:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  402172:	00 
  402173:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  402177:	48 89 57 20          	mov    %rdx,0x20(%rdi)
  40217b:	48 8b 94 24 18 01 00 	mov    0x118(%rsp),%rdx
  402182:	00 
  402183:	48 89 47 10          	mov    %rax,0x10(%rdi)
  402187:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  40218e:	00 
  40218f:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  402194:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  402198:	48 8b 94 24 20 01 00 	mov    0x120(%rsp),%rdx
  40219f:	00 
  4021a0:	48 85 c0             	test   %rax,%rax
  4021a3:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  4021a7:	48 8b 94 24 28 01 00 	mov    0x128(%rsp),%rdx
  4021ae:	00 
  4021af:	48 89 47 18          	mov    %rax,0x18(%rdi)
  4021b3:	48 89 57 38          	mov    %rdx,0x38(%rdi)
  4021b7:	48 8d 57 08          	lea    0x8(%rdi),%rdx
  4021bb:	48 89 57 40          	mov    %rdx,0x40(%rdi)
  4021bf:	48 8d 57 50          	lea    0x50(%rdi),%rdx
  4021c3:	48 89 57 48          	mov    %rdx,0x48(%rdi)
  4021c7:	74 04                	je     4021cd <_Z11encodeImagePfS_S_+0x38d>
  4021c9:	f0 83 00 01          	lock addl $0x1,(%rax)
  4021cd:	83 bc 24 f4 00 00 00 	cmpl   $0x2,0xf4(%rsp)
  4021d4:	02 
  4021d5:	0f 8f 8d 03 00 00    	jg     402568 <_Z11encodeImagePfS_S_+0x728>
  4021db:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
  4021e2:	00 
  4021e3:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4021e7:	48 8b 12             	mov    (%rdx),%rdx
  4021ea:	48 89 10             	mov    %rdx,(%rax)
  4021ed:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
  4021f4:	00 
  4021f5:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4021f9:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  4021fe:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  402202:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402206:	48 83 c7 60          	add    $0x60,%rdi
  40220a:	48 89 7c 24 38       	mov    %rdi,0x38(%rsp)
  40220f:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  402216:	00 
  402217:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40221c:	e8 7f ee ff ff       	callq  4010a0 <_ZN2cv12_OutputArrayC1ERNS_3MatE@plt>
  402221:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  402226:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  40222b:	e8 00 ef ff ff       	callq  401130 <_ZN2cv5mergeERKSt6vectorINS_3MatESaIS1_EERKNS_12_OutputArrayE@plt>
  402230:	48 8d b4 24 90 00 00 	lea    0x90(%rsp),%rsi
  402237:	00 
  402238:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40223d:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402244:	00 00 
  402246:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  40224d:	00 00 
  40224f:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  402256:	00 00 
  402258:	e8 c3 ee ff ff       	callq  401120 <_ZN2cv11_InputArrayC1ERKNS_3MatE@plt>
  40225d:	48 8d 54 24 0e       	lea    0xe(%rsp),%rdx
  402262:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402267:	be 90 41 40 00       	mov    $0x404190,%esi
  40226c:	e8 8f ef ff ff       	callq  401200 <_ZNSsC1EPKcRKSaIcE@plt>
  402271:	48 8d 54 24 50       	lea    0x50(%rsp),%rdx
  402276:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  40227b:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  402280:	e8 6b ee ff ff       	callq  4010f0 <_ZN2cv7imwriteERKSsRKNS_11_InputArrayERKSt6vectorIiSaIiEE@plt>
  402285:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40228a:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  40228e:	48 81 ff 20 51 60 00 	cmp    $0x605120,%rdi
  402295:	0f 85 14 04 00 00    	jne    4026af <_Z11encodeImagePfS_S_+0x86f>
  40229b:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  4022a0:	48 85 ff             	test   %rdi,%rdi
  4022a3:	74 05                	je     4022aa <_Z11encodeImagePfS_S_+0x46a>
  4022a5:	e8 26 ee ff ff       	callq  4010d0 <_ZdlPv@plt>
  4022aa:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
  4022b1:	00 
  4022b2:	48 85 c0             	test   %rax,%rax
  4022b5:	74 0a                	je     4022c1 <_Z11encodeImagePfS_S_+0x481>
  4022b7:	f0 83 28 01          	lock subl $0x1,(%rax)
  4022bb:	0f 84 cf 02 00 00    	je     402590 <_Z11encodeImagePfS_S_+0x750>
  4022c1:	48 8b 84 24 f0 01 00 	mov    0x1f0(%rsp),%rax
  4022c8:	00 
  4022c9:	48 8b bc 24 f8 01 00 	mov    0x1f8(%rsp),%rdi
  4022d0:	00 
  4022d1:	48 c7 84 24 e0 01 00 	movq   $0x0,0x1e0(%rsp)
  4022d8:	00 00 00 00 00 
  4022dd:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  4022e4:	00 00 00 00 00 
  4022e9:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  4022f0:	00 00 00 00 00 
  4022f5:	48 c7 84 24 c0 01 00 	movq   $0x0,0x1c0(%rsp)
  4022fc:	00 00 00 00 00 
  402301:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402307:	48 8d 84 24 00 02 00 	lea    0x200(%rsp),%rax
  40230e:	00 
  40230f:	48 c7 84 24 c8 01 00 	movq   $0x0,0x1c8(%rsp)
  402316:	00 00 00 00 00 
  40231b:	48 39 c7             	cmp    %rax,%rdi
  40231e:	74 05                	je     402325 <_Z11encodeImagePfS_S_+0x4e5>
  402320:	e8 6b ee ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402325:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  40232c:	00 
  40232d:	48 85 c0             	test   %rax,%rax
  402330:	74 0a                	je     40233c <_Z11encodeImagePfS_S_+0x4fc>
  402332:	f0 83 28 01          	lock subl $0x1,(%rax)
  402336:	0f 84 8a 02 00 00    	je     4025c6 <_Z11encodeImagePfS_S_+0x786>
  40233c:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
  402343:	00 
  402344:	48 8b bc 24 98 01 00 	mov    0x198(%rsp),%rdi
  40234b:	00 
  40234c:	48 c7 84 24 80 01 00 	movq   $0x0,0x180(%rsp)
  402353:	00 00 00 00 00 
  402358:	48 c7 84 24 78 01 00 	movq   $0x0,0x178(%rsp)
  40235f:	00 00 00 00 00 
  402364:	48 c7 84 24 70 01 00 	movq   $0x0,0x170(%rsp)
  40236b:	00 00 00 00 00 
  402370:	48 c7 84 24 60 01 00 	movq   $0x0,0x160(%rsp)
  402377:	00 00 00 00 00 
  40237c:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402382:	48 8d 84 24 a0 01 00 	lea    0x1a0(%rsp),%rax
  402389:	00 
  40238a:	48 c7 84 24 68 01 00 	movq   $0x0,0x168(%rsp)
  402391:	00 00 00 00 00 
  402396:	48 39 c7             	cmp    %rax,%rdi
  402399:	74 05                	je     4023a0 <_Z11encodeImagePfS_S_+0x560>
  40239b:	e8 f0 ed ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  4023a0:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  4023a7:	00 
  4023a8:	48 85 c0             	test   %rax,%rax
  4023ab:	74 0a                	je     4023b7 <_Z11encodeImagePfS_S_+0x577>
  4023ad:	f0 83 28 01          	lock subl $0x1,(%rax)
  4023b1:	0f 84 fd 01 00 00    	je     4025b4 <_Z11encodeImagePfS_S_+0x774>
  4023b7:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  4023be:	00 
  4023bf:	48 8b bc 24 38 01 00 	mov    0x138(%rsp),%rdi
  4023c6:	00 
  4023c7:	48 c7 84 24 20 01 00 	movq   $0x0,0x120(%rsp)
  4023ce:	00 00 00 00 00 
  4023d3:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  4023da:	00 00 00 00 00 
  4023df:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  4023e6:	00 00 00 00 00 
  4023eb:	48 c7 84 24 00 01 00 	movq   $0x0,0x100(%rsp)
  4023f2:	00 00 00 00 00 
  4023f7:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4023fd:	48 8d 84 24 40 01 00 	lea    0x140(%rsp),%rax
  402404:	00 
  402405:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
  40240c:	00 00 00 00 00 
  402411:	48 39 c7             	cmp    %rax,%rdi
  402414:	74 05                	je     40241b <_Z11encodeImagePfS_S_+0x5db>
  402416:	e8 75 ed ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  40241b:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  402422:	00 
  402423:	48 85 c0             	test   %rax,%rax
  402426:	74 0a                	je     402432 <_Z11encodeImagePfS_S_+0x5f2>
  402428:	f0 83 28 01          	lock subl $0x1,(%rax)
  40242c:	0f 84 70 01 00 00    	je     4025a2 <_Z11encodeImagePfS_S_+0x762>
  402432:	48 8b 84 24 d0 00 00 	mov    0xd0(%rsp),%rax
  402439:	00 
  40243a:	48 8b bc 24 d8 00 00 	mov    0xd8(%rsp),%rdi
  402441:	00 
  402442:	48 c7 84 24 c0 00 00 	movq   $0x0,0xc0(%rsp)
  402449:	00 00 00 00 00 
  40244e:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  402455:	00 00 00 00 00 
  40245a:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  402461:	00 00 00 00 00 
  402466:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  40246d:	00 00 00 00 00 
  402472:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402478:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  40247f:	00 
  402480:	48 c7 84 24 a8 00 00 	movq   $0x0,0xa8(%rsp)
  402487:	00 00 00 00 00 
  40248c:	48 39 c7             	cmp    %rax,%rdi
  40248f:	74 05                	je     402496 <_Z11encodeImagePfS_S_+0x656>
  402491:	e8 fa ec ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402496:	4c 8b 64 24 38       	mov    0x38(%rsp),%r12
  40249b:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  4024a0:	49 39 dc             	cmp    %rbx,%r12
  4024a3:	4c 89 e7             	mov    %r12,%rdi
  4024a6:	75 55                	jne    4024fd <_Z11encodeImagePfS_S_+0x6bd>
  4024a8:	e9 a0 00 00 00       	jmpq   40254d <_Z11encodeImagePfS_S_+0x70d>
  4024ad:	0f 1f 00             	nopl   (%rax)
  4024b0:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4024b4:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4024bb:	00 
  4024bc:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  4024c3:	00 
  4024c4:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  4024cb:	00 
  4024cc:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  4024d3:	00 
  4024d4:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4024da:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  4024de:	48 8d 43 50          	lea    0x50(%rbx),%rax
  4024e2:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  4024e9:	00 
  4024ea:	48 39 c7             	cmp    %rax,%rdi
  4024ed:	74 05                	je     4024f4 <_Z11encodeImagePfS_S_+0x6b4>
  4024ef:	e8 9c ec ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  4024f4:	48 83 c3 60          	add    $0x60,%rbx
  4024f8:	49 39 dc             	cmp    %rbx,%r12
  4024fb:	74 4b                	je     402548 <_Z11encodeImagePfS_S_+0x708>
  4024fd:	48 8b 43 18          	mov    0x18(%rbx),%rax
  402501:	48 85 c0             	test   %rax,%rax
  402504:	74 aa                	je     4024b0 <_Z11encodeImagePfS_S_+0x670>
  402506:	f0 83 28 01          	lock subl $0x1,(%rax)
  40250a:	75 a4                	jne    4024b0 <_Z11encodeImagePfS_S_+0x670>
  40250c:	48 89 df             	mov    %rbx,%rdi
  40250f:	e8 ac eb ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402514:	eb 9a                	jmp    4024b0 <_Z11encodeImagePfS_S_+0x670>
  402516:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40251d:	00 00 00 
  402520:	48 8d b4 24 50 01 00 	lea    0x150(%rsp),%rsi
  402527:	00 
  402528:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  40252f:	e8 2c eb ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402534:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402539:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40253e:	e9 f8 fb ff ff       	jmpq   40213b <_Z11encodeImagePfS_S_+0x2fb>
  402543:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402548:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40254d:	48 85 ff             	test   %rdi,%rdi
  402550:	74 05                	je     402557 <_Z11encodeImagePfS_S_+0x717>
  402552:	e8 79 eb ff ff       	callq  4010d0 <_ZdlPv@plt>
  402557:	48 8d 65 f0          	lea    -0x10(%rbp),%rsp
  40255b:	31 c0                	xor    %eax,%eax
  40255d:	5b                   	pop    %rbx
  40255e:	41 5c                	pop    %r12
  402560:	5d                   	pop    %rbp
  402561:	c3                   	retq   
  402562:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402568:	48 8d b4 24 f0 00 00 	lea    0xf0(%rsp),%rsi
  40256f:	00 
  402570:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  402577:	e8 e4 ea ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  40257c:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402581:	e9 80 fc ff ff       	jmpq   402206 <_Z11encodeImagePfS_S_+0x3c6>
  402586:	48 89 c6             	mov    %rax,%rsi
  402589:	31 ff                	xor    %edi,%edi
  40258b:	e9 ab fb ff ff       	jmpq   40213b <_Z11encodeImagePfS_S_+0x2fb>
  402590:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  402597:	00 
  402598:	e8 23 eb ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  40259d:	e9 1f fd ff ff       	jmpq   4022c1 <_Z11encodeImagePfS_S_+0x481>
  4025a2:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  4025a9:	00 
  4025aa:	e8 11 eb ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  4025af:	e9 7e fe ff ff       	jmpq   402432 <_Z11encodeImagePfS_S_+0x5f2>
  4025b4:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  4025bb:	00 
  4025bc:	e8 ff ea ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  4025c1:	e9 f1 fd ff ff       	jmpq   4023b7 <_Z11encodeImagePfS_S_+0x577>
  4025c6:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  4025cd:	00 
  4025ce:	e8 ed ea ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  4025d3:	e9 64 fd ff ff       	jmpq   40233c <_Z11encodeImagePfS_S_+0x4fc>
  4025d8:	48 8d 94 24 f0 00 00 	lea    0xf0(%rsp),%rdx
  4025df:	00 
  4025e0:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4025e5:	e8 56 0b 00 00       	callq  403140 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  4025ea:	e9 20 fc ff ff       	jmpq   40220f <_Z11encodeImagePfS_S_+0x3cf>
  4025ef:	48 89 fe             	mov    %rdi,%rsi
  4025f2:	48 8d 94 24 50 01 00 	lea    0x150(%rsp),%rdx
  4025f9:	00 
  4025fa:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  4025ff:	e8 3c 0b 00 00       	callq  403140 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>
  402604:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
  402609:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  40260e:	e9 31 fb ff ff       	jmpq   402144 <_Z11encodeImagePfS_S_+0x304>
  402613:	48 89 c3             	mov    %rax,%rbx
  402616:	c5 f8 77             	vzeroupper 
  402619:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  402620:	00 
  402621:	e8 ba 09 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402626:	48 8d bc 24 50 01 00 	lea    0x150(%rsp),%rdi
  40262d:	00 
  40262e:	e8 ad 09 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402633:	48 8d bc 24 f0 00 00 	lea    0xf0(%rsp),%rdi
  40263a:	00 
  40263b:	e8 a0 09 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402640:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  402647:	00 
  402648:	e8 93 09 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  40264d:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  402652:	e8 09 0a 00 00       	callq  403060 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>
  402657:	48 89 df             	mov    %rbx,%rdi
  40265a:	e8 a1 ea ff ff       	callq  401100 <_Unwind_Resume@plt>
  40265f:	48 89 c3             	mov    %rax,%rbx
  402662:	c5 f8 77             	vzeroupper 
  402665:	eb e6                	jmp    40264d <_Z11encodeImagePfS_S_+0x80d>
  402667:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40266c:	48 89 c3             	mov    %rax,%rbx
  40266f:	48 85 ff             	test   %rdi,%rdi
  402672:	74 69                	je     4026dd <_Z11encodeImagePfS_S_+0x89d>
  402674:	c5 f8 77             	vzeroupper 
  402677:	e8 54 ea ff ff       	callq  4010d0 <_ZdlPv@plt>
  40267c:	eb d9                	jmp    402657 <_Z11encodeImagePfS_S_+0x817>
  40267e:	48 89 c3             	mov    %rax,%rbx
  402681:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  402686:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  40268b:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  40268f:	c5 f8 77             	vzeroupper 
  402692:	e8 b9 ea ff ff       	callq  401150 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  402697:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
  40269c:	48 85 ff             	test   %rdi,%rdi
  40269f:	0f 84 74 ff ff ff    	je     402619 <_Z11encodeImagePfS_S_+0x7d9>
  4026a5:	e8 26 ea ff ff       	callq  4010d0 <_ZdlPv@plt>
  4026aa:	e9 6a ff ff ff       	jmpq   402619 <_Z11encodeImagePfS_S_+0x7d9>
  4026af:	b9 a0 11 40 00       	mov    $0x4011a0,%ecx
  4026b4:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  4026b8:	48 85 c9             	test   %rcx,%rcx
  4026bb:	74 28                	je     4026e5 <_Z11encodeImagePfS_S_+0x8a5>
  4026bd:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4026c2:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  4026c6:	85 c0                	test   %eax,%eax
  4026c8:	0f 8f cd fb ff ff    	jg     40229b <_Z11encodeImagePfS_S_+0x45b>
  4026ce:	48 8d 74 24 0f       	lea    0xf(%rsp),%rsi
  4026d3:	e8 e8 ea ff ff       	callq  4011c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  4026d8:	e9 be fb ff ff       	jmpq   40229b <_Z11encodeImagePfS_S_+0x45b>
  4026dd:	c5 f8 77             	vzeroupper 
  4026e0:	e9 72 ff ff ff       	jmpq   402657 <_Z11encodeImagePfS_S_+0x817>
  4026e5:	8b 50 f8             	mov    -0x8(%rax),%edx
  4026e8:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  4026eb:	89 48 f8             	mov    %ecx,-0x8(%rax)
  4026ee:	89 d0                	mov    %edx,%eax
  4026f0:	eb d4                	jmp    4026c6 <_Z11encodeImagePfS_S_+0x886>
  4026f2:	48 89 c3             	mov    %rax,%rbx
  4026f5:	c5 f8 77             	vzeroupper 
  4026f8:	eb 9d                	jmp    402697 <_Z11encodeImagePfS_S_+0x857>
  4026fa:	48 89 c3             	mov    %rax,%rbx
  4026fd:	c5 f8 77             	vzeroupper 
  402700:	e9 21 ff ff ff       	jmpq   402626 <_Z11encodeImagePfS_S_+0x7e6>
  402705:	48 89 c3             	mov    %rax,%rbx
  402708:	c5 f8 77             	vzeroupper 
  40270b:	e9 23 ff ff ff       	jmpq   402633 <_Z11encodeImagePfS_S_+0x7f3>
  402710:	48 89 c3             	mov    %rax,%rbx
  402713:	c5 f8 77             	vzeroupper 
  402716:	e9 25 ff ff ff       	jmpq   402640 <_Z11encodeImagePfS_S_+0x800>
  40271b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000402720 <_Z11decodeImagePfS_S_>:
  402720:	55                   	push   %rbp
  402721:	48 89 e5             	mov    %rsp,%rbp
  402724:	41 57                	push   %r15
  402726:	41 56                	push   %r14
  402728:	41 55                	push   %r13
  40272a:	49 89 d5             	mov    %rdx,%r13
  40272d:	41 54                	push   %r12
  40272f:	49 89 f4             	mov    %rsi,%r12
  402732:	be d8 41 40 00       	mov    $0x4041d8,%esi
  402737:	53                   	push   %rbx
  402738:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40273c:	48 81 ec a0 02 00 00 	sub    $0x2a0,%rsp
  402743:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
  402748:	48 8d 9c 24 80 01 00 	lea    0x180(%rsp),%rbx
  40274f:	00 
  402750:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  402755:	48 c7 84 24 b8 00 00 	movq   $0x0,0xb8(%rsp)
  40275c:	00 00 00 00 00 
  402761:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  402768:	00 00 00 00 00 
  40276d:	48 8d 54 24 1e       	lea    0x1e(%rsp),%rdx
  402772:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  402779:	00 
  40277a:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  402781:	00 
  402782:	c7 44 24 60 00 00 ff 	movl   $0x42ff0000,0x60(%rsp)
  402789:	42 
  40278a:	c7 44 24 6c 00 00 00 	movl   $0x0,0x6c(%rsp)
  402791:	00 
  402792:	c7 44 24 68 00 00 00 	movl   $0x0,0x68(%rsp)
  402799:	00 
  40279a:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
  40279f:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  4027a6:	00 
  4027a7:	48 8d 84 24 c8 00 00 	lea    0xc8(%rsp),%rax
  4027ae:	00 
  4027af:	c7 44 24 64 00 00 00 	movl   $0x0,0x64(%rsp)
  4027b6:	00 
  4027b7:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  4027be:	00 00 00 00 00 
  4027c3:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  4027ca:	00 00 00 00 00 
  4027cf:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  4027d6:	00 
  4027d7:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  4027de:	00 
  4027df:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  4027e6:	00 00 00 00 00 
  4027eb:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  4027f2:	00 00 
  4027f4:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  4027fb:	00 00 
  4027fd:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  402804:	00 
  402805:	48 8d 43 08          	lea    0x8(%rbx),%rax
  402809:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
  402810:	00 00 00 00 00 
  402815:	48 c7 84 24 18 01 00 	movq   $0x0,0x118(%rsp)
  40281c:	00 00 00 00 00 
  402821:	48 c7 84 24 10 01 00 	movq   $0x0,0x110(%rsp)
  402828:	00 00 00 00 00 
  40282d:	c7 84 24 c0 00 00 00 	movl   $0x42ff0000,0xc0(%rsp)
  402834:	00 00 ff 42 
  402838:	c7 84 24 cc 00 00 00 	movl   $0x0,0xcc(%rsp)
  40283f:	00 00 00 00 
  402843:	c7 84 24 c8 00 00 00 	movl   $0x0,0xc8(%rsp)
  40284a:	00 00 00 00 
  40284e:	c7 84 24 c4 00 00 00 	movl   $0x0,0xc4(%rsp)
  402855:	00 00 00 00 
  402859:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402860:	00 00 00 00 00 
  402865:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  40286c:	00 00 00 00 00 
  402871:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  402878:	00 00 00 00 00 
  40287d:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  402884:	00 00 00 00 00 
  402889:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
  402890:	00 
  402891:	48 8d 43 50          	lea    0x50(%rbx),%rax
  402895:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  40289c:	00 00 00 00 00 
  4028a1:	48 c7 84 24 f8 00 00 	movq   $0x0,0xf8(%rsp)
  4028a8:	00 00 00 00 00 
  4028ad:	48 c7 84 24 d8 01 00 	movq   $0x0,0x1d8(%rsp)
  4028b4:	00 00 00 00 00 
  4028b9:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
  4028c0:	00 
  4028c1:	48 8d 43 68          	lea    0x68(%rbx),%rax
  4028c5:	48 c7 84 24 d0 01 00 	movq   $0x0,0x1d0(%rsp)
  4028cc:	00 00 00 00 00 
  4028d1:	c7 84 24 80 01 00 00 	movl   $0x42ff0000,0x180(%rsp)
  4028d8:	00 00 ff 42 
  4028dc:	c7 84 24 8c 01 00 00 	movl   $0x0,0x18c(%rsp)
  4028e3:	00 00 00 00 
  4028e7:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
  4028ee:	00 
  4028ef:	48 8d 83 b0 00 00 00 	lea    0xb0(%rbx),%rax
  4028f6:	c7 84 24 88 01 00 00 	movl   $0x0,0x188(%rsp)
  4028fd:	00 00 00 00 
  402901:	c7 84 24 84 01 00 00 	movl   $0x0,0x184(%rsp)
  402908:	00 00 00 00 
  40290c:	48 c7 84 24 b0 01 00 	movq   $0x0,0x1b0(%rsp)
  402913:	00 00 00 00 00 
  402918:	48 89 84 24 28 02 00 	mov    %rax,0x228(%rsp)
  40291f:	00 
  402920:	48 8d 83 c8 00 00 00 	lea    0xc8(%rbx),%rax
  402927:	48 c7 84 24 a8 01 00 	movq   $0x0,0x1a8(%rsp)
  40292e:	00 00 00 00 00 
  402933:	48 c7 84 24 a0 01 00 	movq   $0x0,0x1a0(%rsp)
  40293a:	00 00 00 00 00 
  40293f:	48 c7 84 24 90 01 00 	movq   $0x0,0x190(%rsp)
  402946:	00 00 00 00 00 
  40294b:	48 89 84 24 80 02 00 	mov    %rax,0x280(%rsp)
  402952:	00 
  402953:	48 8d 83 10 01 00 00 	lea    0x110(%rbx),%rax
  40295a:	48 c7 84 24 98 01 00 	movq   $0x0,0x198(%rsp)
  402961:	00 00 00 00 00 
  402966:	48 c7 84 24 b8 01 00 	movq   $0x0,0x1b8(%rsp)
  40296d:	00 00 00 00 00 
  402972:	48 c7 84 24 38 02 00 	movq   $0x0,0x238(%rsp)
  402979:	00 00 00 00 00 
  40297e:	48 c7 84 24 30 02 00 	movq   $0x0,0x230(%rsp)
  402985:	00 00 00 00 00 
  40298a:	c7 84 24 e0 01 00 00 	movl   $0x42ff0000,0x1e0(%rsp)
  402991:	00 00 ff 42 
  402995:	c7 84 24 ec 01 00 00 	movl   $0x0,0x1ec(%rsp)
  40299c:	00 00 00 00 
  4029a0:	c7 84 24 e8 01 00 00 	movl   $0x0,0x1e8(%rsp)
  4029a7:	00 00 00 00 
  4029ab:	c7 84 24 e4 01 00 00 	movl   $0x0,0x1e4(%rsp)
  4029b2:	00 00 00 00 
  4029b6:	48 c7 84 24 10 02 00 	movq   $0x0,0x210(%rsp)
  4029bd:	00 00 00 00 00 
  4029c2:	48 c7 84 24 08 02 00 	movq   $0x0,0x208(%rsp)
  4029c9:	00 00 00 00 00 
  4029ce:	48 c7 84 24 00 02 00 	movq   $0x0,0x200(%rsp)
  4029d5:	00 00 00 00 00 
  4029da:	48 c7 84 24 f0 01 00 	movq   $0x0,0x1f0(%rsp)
  4029e1:	00 00 00 00 00 
  4029e6:	48 c7 84 24 f8 01 00 	movq   $0x0,0x1f8(%rsp)
  4029ed:	00 00 00 00 00 
  4029f2:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
  4029f9:	00 00 00 00 00 
  4029fe:	48 89 84 24 88 02 00 	mov    %rax,0x288(%rsp)
  402a05:	00 
  402a06:	48 c7 84 24 98 02 00 	movq   $0x0,0x298(%rsp)
  402a0d:	00 00 00 00 00 
  402a12:	48 c7 84 24 90 02 00 	movq   $0x0,0x290(%rsp)
  402a19:	00 00 00 00 00 
  402a1e:	c7 84 24 40 02 00 00 	movl   $0x42ff0000,0x240(%rsp)
  402a25:	00 00 ff 42 
  402a29:	c7 84 24 4c 02 00 00 	movl   $0x0,0x24c(%rsp)
  402a30:	00 00 00 00 
  402a34:	c7 84 24 48 02 00 00 	movl   $0x0,0x248(%rsp)
  402a3b:	00 00 00 00 
  402a3f:	c7 84 24 44 02 00 00 	movl   $0x0,0x244(%rsp)
  402a46:	00 00 00 00 
  402a4a:	48 c7 84 24 70 02 00 	movq   $0x0,0x270(%rsp)
  402a51:	00 00 00 00 00 
  402a56:	48 c7 84 24 68 02 00 	movq   $0x0,0x268(%rsp)
  402a5d:	00 00 00 00 00 
  402a62:	48 c7 84 24 60 02 00 	movq   $0x0,0x260(%rsp)
  402a69:	00 00 00 00 00 
  402a6e:	48 c7 84 24 50 02 00 	movq   $0x0,0x250(%rsp)
  402a75:	00 00 00 00 00 
  402a7a:	48 c7 84 24 58 02 00 	movq   $0x0,0x258(%rsp)
  402a81:	00 00 00 00 00 
  402a86:	48 c7 84 24 78 02 00 	movq   $0x0,0x278(%rsp)
  402a8d:	00 00 00 00 00 
  402a92:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  402a99:	00 00 
  402a9b:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  402aa2:	00 00 
  402aa4:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  402aab:	00 00 
  402aad:	e8 4e e7 ff ff       	callq  401200 <_ZNSsC1EPKcRKSaIcE@plt>
  402ab2:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  402ab7:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  402abe:	00 
  402abf:	ba 01 00 00 00       	mov    $0x1,%edx
  402ac4:	e8 27 e7 ff ff       	callq  4011f0 <_ZN2cv6imreadERKSsi@plt>
  402ac9:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  402ad0:	00 
  402ad1:	48 85 c0             	test   %rax,%rax
  402ad4:	74 04                	je     402ada <_Z11decodeImagePfS_S_+0x3ba>
  402ad6:	f0 83 00 01          	lock addl $0x1,(%rax)
  402ada:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  402adf:	48 85 c0             	test   %rax,%rax
  402ae2:	74 0a                	je     402aee <_Z11decodeImagePfS_S_+0x3ce>
  402ae4:	f0 83 28 01          	lock subl $0x1,(%rax)
  402ae8:	0f 84 d2 03 00 00    	je     402ec0 <_Z11decodeImagePfS_S_+0x7a0>
  402aee:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  402af5:	00 
  402af6:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  402afd:	00 00 00 00 00 
  402b02:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  402b09:	00 00 00 00 00 
  402b0e:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402b15:	00 00 00 00 00 
  402b1a:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402b21:	00 00 
  402b23:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402b29:	83 7c 24 64 02       	cmpl   $0x2,0x64(%rsp)
  402b2e:	8b 84 24 20 01 00 00 	mov    0x120(%rsp),%eax
  402b35:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  402b3c:	00 00 
  402b3e:	89 44 24 60          	mov    %eax,0x60(%rsp)
  402b42:	7f 10                	jg     402b54 <_Z11decodeImagePfS_S_+0x434>
  402b44:	8b 84 24 24 01 00 00 	mov    0x124(%rsp),%eax
  402b4b:	83 f8 02             	cmp    $0x2,%eax
  402b4e:	0f 8e 1c 03 00 00    	jle    402e70 <_Z11decodeImagePfS_S_+0x750>
  402b54:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  402b5b:	00 
  402b5c:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402b61:	e8 fa e4 ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  402b66:	c5 f9 6f 84 24 30 01 	vmovdqa 0x130(%rsp),%xmm0
  402b6d:	00 00 
  402b6f:	48 8b 84 24 38 01 00 	mov    0x138(%rsp),%rax
  402b76:	00 
  402b77:	c5 f9 7f 44 24 70    	vmovdqa %xmm0,0x70(%rsp)
  402b7d:	48 85 c0             	test   %rax,%rax
  402b80:	c5 f9 6f 84 24 40 01 	vmovdqa 0x140(%rsp),%xmm0
  402b87:	00 00 
  402b89:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  402b90:	00 00 
  402b92:	c5 f9 6f 84 24 50 01 	vmovdqa 0x150(%rsp),%xmm0
  402b99:	00 00 
  402b9b:	c5 f9 7f 84 24 90 00 	vmovdqa %xmm0,0x90(%rsp)
  402ba2:	00 00 
  402ba4:	74 0a                	je     402bb0 <_Z11decodeImagePfS_S_+0x490>
  402ba6:	f0 83 28 01          	lock subl $0x1,(%rax)
  402baa:	0f 84 20 03 00 00    	je     402ed0 <_Z11decodeImagePfS_S_+0x7b0>
  402bb0:	48 8b 84 24 60 01 00 	mov    0x160(%rsp),%rax
  402bb7:	00 
  402bb8:	48 8b bc 24 68 01 00 	mov    0x168(%rsp),%rdi
  402bbf:	00 
  402bc0:	48 c7 84 24 50 01 00 	movq   $0x0,0x150(%rsp)
  402bc7:	00 00 00 00 00 
  402bcc:	48 c7 84 24 48 01 00 	movq   $0x0,0x148(%rsp)
  402bd3:	00 00 00 00 00 
  402bd8:	48 c7 84 24 40 01 00 	movq   $0x0,0x140(%rsp)
  402bdf:	00 00 00 00 00 
  402be4:	48 c7 84 24 30 01 00 	movq   $0x0,0x130(%rsp)
  402beb:	00 00 00 00 00 
  402bf0:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402bf6:	48 8d 84 24 70 01 00 	lea    0x170(%rsp),%rax
  402bfd:	00 
  402bfe:	48 c7 84 24 38 01 00 	movq   $0x0,0x138(%rsp)
  402c05:	00 00 00 00 00 
  402c0a:	48 39 c7             	cmp    %rax,%rdi
  402c0d:	74 05                	je     402c14 <_Z11decodeImagePfS_S_+0x4f4>
  402c0f:	e8 7c e5 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402c14:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  402c19:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402c1d:	48 81 ff 20 51 60 00 	cmp    $0x605120,%rdi
  402c24:	0f 85 e5 02 00 00    	jne    402f0f <_Z11decodeImagePfS_S_+0x7ef>
  402c2a:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402c2f:	48 89 de             	mov    %rbx,%rsi
  402c32:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
  402c37:	44 8b 7c 24 6c       	mov    0x6c(%rsp),%r15d
  402c3c:	e8 9f e5 ff ff       	callq  4011e0 <_ZN2cv5splitERKNS_3MatEPS0_@plt>
  402c41:	48 8b 94 24 a8 01 00 	mov    0x1a8(%rsp),%rdx
  402c48:	00 
  402c49:	48 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%rsi
  402c50:	00 
  402c51:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402c56:	c6 04 24 00          	movb   $0x0,(%rsp)
  402c5a:	e8 91 0e 00 00       	callq  403af0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402c5f:	45 0f af f7          	imul   %r15d,%r14d
  402c63:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402c68:	4c 89 ef             	mov    %r13,%rdi
  402c6b:	4d 63 f6             	movslq %r14d,%r14
  402c6e:	49 c1 e6 02          	shl    $0x2,%r14
  402c72:	4c 89 f2             	mov    %r14,%rdx
  402c75:	e8 d6 e3 ff ff       	callq  401050 <memcpy@plt>
  402c7a:	48 8b 94 24 08 02 00 	mov    0x208(%rsp),%rdx
  402c81:	00 
  402c82:	48 8b b4 24 00 02 00 	mov    0x200(%rsp),%rsi
  402c89:	00 
  402c8a:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402c8f:	c6 04 24 00          	movb   $0x0,(%rsp)
  402c93:	e8 58 0e 00 00       	callq  403af0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402c98:	48 8b 74 24 40       	mov    0x40(%rsp),%rsi
  402c9d:	4c 89 f2             	mov    %r14,%rdx
  402ca0:	4c 89 e7             	mov    %r12,%rdi
  402ca3:	e8 a8 e3 ff ff       	callq  401050 <memcpy@plt>
  402ca8:	48 8b 94 24 68 02 00 	mov    0x268(%rsp),%rdx
  402caf:	00 
  402cb0:	48 8b b4 24 60 02 00 	mov    0x260(%rsp),%rsi
  402cb7:	00 
  402cb8:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  402cbd:	c6 04 24 00          	movb   $0x0,(%rsp)
  402cc1:	e8 2a 0e 00 00       	callq  403af0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>
  402cc6:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
  402ccb:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  402cd0:	4c 89 f2             	mov    %r14,%rdx
  402cd3:	4c 89 e6             	mov    %r12,%rsi
  402cd6:	e8 75 e3 ff ff       	callq  401050 <memcpy@plt>
  402cdb:	4d 85 e4             	test   %r12,%r12
  402cde:	74 08                	je     402ce8 <_Z11decodeImagePfS_S_+0x5c8>
  402ce0:	4c 89 e7             	mov    %r12,%rdi
  402ce3:	e8 e8 e3 ff ff       	callq  4010d0 <_ZdlPv@plt>
  402ce8:	4c 8d a4 24 a0 02 00 	lea    0x2a0(%rsp),%r12
  402cef:	00 
  402cf0:	49 83 ec 60          	sub    $0x60,%r12
  402cf4:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  402cf9:	48 85 c0             	test   %rax,%rax
  402cfc:	74 0a                	je     402d08 <_Z11decodeImagePfS_S_+0x5e8>
  402cfe:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d02:	0f 84 58 01 00 00    	je     402e60 <_Z11decodeImagePfS_S_+0x740>
  402d08:	49 8b 44 24 40       	mov    0x40(%r12),%rax
  402d0d:	49 c7 44 24 30 00 00 	movq   $0x0,0x30(%r12)
  402d14:	00 00 
  402d16:	49 c7 44 24 28 00 00 	movq   $0x0,0x28(%r12)
  402d1d:	00 00 
  402d1f:	49 c7 44 24 20 00 00 	movq   $0x0,0x20(%r12)
  402d26:	00 00 
  402d28:	49 c7 44 24 10 00 00 	movq   $0x0,0x10(%r12)
  402d2f:	00 00 
  402d31:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402d37:	49 8b 7c 24 48       	mov    0x48(%r12),%rdi
  402d3c:	49 8d 44 24 50       	lea    0x50(%r12),%rax
  402d41:	49 c7 44 24 18 00 00 	movq   $0x0,0x18(%r12)
  402d48:	00 00 
  402d4a:	48 39 c7             	cmp    %rax,%rdi
  402d4d:	74 05                	je     402d54 <_Z11decodeImagePfS_S_+0x634>
  402d4f:	e8 3c e4 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402d54:	49 39 dc             	cmp    %rbx,%r12
  402d57:	75 97                	jne    402cf0 <_Z11decodeImagePfS_S_+0x5d0>
  402d59:	48 8b 84 24 d8 00 00 	mov    0xd8(%rsp),%rax
  402d60:	00 
  402d61:	48 85 c0             	test   %rax,%rax
  402d64:	74 0a                	je     402d70 <_Z11decodeImagePfS_S_+0x650>
  402d66:	f0 83 28 01          	lock subl $0x1,(%rax)
  402d6a:	0f 84 78 01 00 00    	je     402ee8 <_Z11decodeImagePfS_S_+0x7c8>
  402d70:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  402d77:	00 
  402d78:	48 8b bc 24 08 01 00 	mov    0x108(%rsp),%rdi
  402d7f:	00 
  402d80:	48 c7 84 24 f0 00 00 	movq   $0x0,0xf0(%rsp)
  402d87:	00 00 00 00 00 
  402d8c:	48 c7 84 24 e8 00 00 	movq   $0x0,0xe8(%rsp)
  402d93:	00 00 00 00 00 
  402d98:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  402d9f:	00 00 00 00 00 
  402da4:	48 c7 84 24 d0 00 00 	movq   $0x0,0xd0(%rsp)
  402dab:	00 00 00 00 00 
  402db0:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402db6:	48 8d 84 24 10 01 00 	lea    0x110(%rsp),%rax
  402dbd:	00 
  402dbe:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  402dc5:	00 00 00 00 00 
  402dca:	48 39 c7             	cmp    %rax,%rdi
  402dcd:	74 05                	je     402dd4 <_Z11decodeImagePfS_S_+0x6b4>
  402dcf:	e8 bc e3 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402dd4:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  402dd9:	48 85 c0             	test   %rax,%rax
  402ddc:	74 0a                	je     402de8 <_Z11decodeImagePfS_S_+0x6c8>
  402dde:	f0 83 28 01          	lock subl $0x1,(%rax)
  402de2:	0f 84 18 01 00 00    	je     402f00 <_Z11decodeImagePfS_S_+0x7e0>
  402de8:	48 8b 84 24 a0 00 00 	mov    0xa0(%rsp),%rax
  402def:	00 
  402df0:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  402df7:	00 
  402df8:	48 c7 84 24 90 00 00 	movq   $0x0,0x90(%rsp)
  402dff:	00 00 00 00 00 
  402e04:	48 c7 84 24 88 00 00 	movq   $0x0,0x88(%rsp)
  402e0b:	00 00 00 00 00 
  402e10:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  402e17:	00 00 00 00 00 
  402e1c:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  402e23:	00 00 
  402e25:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  402e2b:	48 8d 84 24 b0 00 00 	lea    0xb0(%rsp),%rax
  402e32:	00 
  402e33:	48 c7 44 24 78 00 00 	movq   $0x0,0x78(%rsp)
  402e3a:	00 00 
  402e3c:	48 39 c7             	cmp    %rax,%rdi
  402e3f:	74 05                	je     402e46 <_Z11decodeImagePfS_S_+0x726>
  402e41:	e8 4a e3 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  402e46:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  402e4a:	31 c0                	xor    %eax,%eax
  402e4c:	5b                   	pop    %rbx
  402e4d:	41 5c                	pop    %r12
  402e4f:	41 5d                	pop    %r13
  402e51:	41 5e                	pop    %r14
  402e53:	41 5f                	pop    %r15
  402e55:	5d                   	pop    %rbp
  402e56:	c3                   	retq   
  402e57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  402e5e:	00 00 
  402e60:	4c 89 e7             	mov    %r12,%rdi
  402e63:	e8 58 e2 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402e68:	e9 9b fe ff ff       	jmpq   402d08 <_Z11decodeImagePfS_S_+0x5e8>
  402e6d:	0f 1f 00             	nopl   (%rax)
  402e70:	89 44 24 64          	mov    %eax,0x64(%rsp)
  402e74:	8b 84 24 28 01 00 00 	mov    0x128(%rsp),%eax
  402e7b:	89 44 24 68          	mov    %eax,0x68(%rsp)
  402e7f:	8b 84 24 2c 01 00 00 	mov    0x12c(%rsp),%eax
  402e86:	89 44 24 6c          	mov    %eax,0x6c(%rsp)
  402e8a:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402e91:	00 
  402e92:	48 8b 10             	mov    (%rax),%rdx
  402e95:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  402e9c:	00 
  402e9d:	48 89 10             	mov    %rdx,(%rax)
  402ea0:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  402ea7:	00 
  402ea8:	48 8b 50 08          	mov    0x8(%rax),%rdx
  402eac:	48 8b 84 24 a8 00 00 	mov    0xa8(%rsp),%rax
  402eb3:	00 
  402eb4:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402eb8:	e9 a9 fc ff ff       	jmpq   402b66 <_Z11decodeImagePfS_S_+0x446>
  402ebd:	0f 1f 00             	nopl   (%rax)
  402ec0:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402ec5:	e8 f6 e1 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402eca:	e9 1f fc ff ff       	jmpq   402aee <_Z11decodeImagePfS_S_+0x3ce>
  402ecf:	90                   	nop
  402ed0:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  402ed7:	00 
  402ed8:	e8 e3 e1 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402edd:	e9 ce fc ff ff       	jmpq   402bb0 <_Z11decodeImagePfS_S_+0x490>
  402ee2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402ee8:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  402eef:	00 
  402ef0:	e8 cb e1 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402ef5:	e9 76 fe ff ff       	jmpq   402d70 <_Z11decodeImagePfS_S_+0x650>
  402efa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  402f00:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402f05:	e8 b6 e1 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  402f0a:	e9 d9 fe ff ff       	jmpq   402de8 <_Z11decodeImagePfS_S_+0x6c8>
  402f0f:	b9 a0 11 40 00       	mov    $0x4011a0,%ecx
  402f14:	48 8d 57 10          	lea    0x10(%rdi),%rdx
  402f18:	48 85 c9             	test   %rcx,%rcx
  402f1b:	74 46                	je     402f63 <_Z11decodeImagePfS_S_+0x843>
  402f1d:	83 c8 ff             	or     $0xffffffff,%eax
  402f20:	f0 0f c1 02          	lock xadd %eax,(%rdx)
  402f24:	85 c0                	test   %eax,%eax
  402f26:	0f 8f fe fc ff ff    	jg     402c2a <_Z11decodeImagePfS_S_+0x50a>
  402f2c:	48 8d b4 24 20 01 00 	lea    0x120(%rsp),%rsi
  402f33:	00 
  402f34:	e8 87 e2 ff ff       	callq  4011c0 <_ZNSs4_Rep10_M_destroyERKSaIcE@plt>
  402f39:	e9 ec fc ff ff       	jmpq   402c2a <_Z11decodeImagePfS_S_+0x50a>
  402f3e:	48 89 c3             	mov    %rax,%rbx
  402f41:	c5 f8 77             	vzeroupper 
  402f44:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  402f4b:	00 
  402f4c:	e8 8f 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402f51:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
  402f56:	e8 85 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402f5b:	48 89 df             	mov    %rbx,%rdi
  402f5e:	e8 9d e1 ff ff       	callq  401100 <_Unwind_Resume@plt>
  402f63:	8b 50 f8             	mov    -0x8(%rax),%edx
  402f66:	8d 4a ff             	lea    -0x1(%rdx),%ecx
  402f69:	89 48 f8             	mov    %ecx,-0x8(%rax)
  402f6c:	89 d0                	mov    %edx,%eax
  402f6e:	eb b4                	jmp    402f24 <_Z11decodeImagePfS_S_+0x804>
  402f70:	49 89 c4             	mov    %rax,%r12
  402f73:	c5 f8 77             	vzeroupper 
  402f76:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  402f7b:	48 8d 74 24 1f       	lea    0x1f(%rsp),%rsi
  402f80:	48 8d 78 e8          	lea    -0x18(%rax),%rdi
  402f84:	e8 c7 e1 ff ff       	callq  401150 <_ZNSs4_Rep10_M_disposeERKSaIcE@plt>
  402f89:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  402f8e:	48 85 ff             	test   %rdi,%rdi
  402f91:	74 05                	je     402f98 <_Z11decodeImagePfS_S_+0x878>
  402f93:	e8 38 e1 ff ff       	callq  4010d0 <_ZdlPv@plt>
  402f98:	48 8d bb c0 00 00 00 	lea    0xc0(%rbx),%rdi
  402f9f:	e8 3c 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402fa4:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  402fa8:	e8 33 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402fad:	48 89 df             	mov    %rbx,%rdi
  402fb0:	4c 89 e3             	mov    %r12,%rbx
  402fb3:	e8 28 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402fb8:	eb 8a                	jmp    402f44 <_Z11decodeImagePfS_S_+0x824>
  402fba:	49 89 c4             	mov    %rax,%r12
  402fbd:	c5 f8 77             	vzeroupper 
  402fc0:	eb c7                	jmp    402f89 <_Z11decodeImagePfS_S_+0x869>
  402fc2:	48 89 c3             	mov    %rax,%rbx
  402fc5:	c5 f8 77             	vzeroupper 
  402fc8:	eb 87                	jmp    402f51 <_Z11decodeImagePfS_S_+0x831>
  402fca:	48 8d bc 24 20 01 00 	lea    0x120(%rsp),%rdi
  402fd1:	00 
  402fd2:	49 89 c4             	mov    %rax,%r12
  402fd5:	c5 f8 77             	vzeroupper 
  402fd8:	e8 03 00 00 00       	callq  402fe0 <_ZN2cv3MatD1Ev>
  402fdd:	eb 97                	jmp    402f76 <_Z11decodeImagePfS_S_+0x856>
  402fdf:	90                   	nop

0000000000402fe0 <_ZN2cv3MatD1Ev>:
  402fe0:	53                   	push   %rbx
  402fe1:	48 8b 47 18          	mov    0x18(%rdi),%rax
  402fe5:	48 89 fb             	mov    %rdi,%rbx
  402fe8:	48 85 c0             	test   %rax,%rax
  402feb:	74 06                	je     402ff3 <_ZN2cv3MatD1Ev+0x13>
  402fed:	f0 83 28 01          	lock subl $0x1,(%rax)
  402ff1:	74 55                	je     403048 <_ZN2cv3MatD1Ev+0x68>
  402ff3:	48 8b 43 40          	mov    0x40(%rbx),%rax
  402ff7:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  402ffb:	48 83 c3 50          	add    $0x50,%rbx
  402fff:	48 c7 43 e0 00 00 00 	movq   $0x0,-0x20(%rbx)
  403006:	00 
  403007:	48 c7 43 d8 00 00 00 	movq   $0x0,-0x28(%rbx)
  40300e:	00 
  40300f:	48 c7 43 d0 00 00 00 	movq   $0x0,-0x30(%rbx)
  403016:	00 
  403017:	48 c7 43 c0 00 00 00 	movq   $0x0,-0x40(%rbx)
  40301e:	00 
  40301f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403025:	48 c7 43 c8 00 00 00 	movq   $0x0,-0x38(%rbx)
  40302c:	00 
  40302d:	48 39 df             	cmp    %rbx,%rdi
  403030:	74 0e                	je     403040 <_ZN2cv3MatD1Ev+0x60>
  403032:	5b                   	pop    %rbx
  403033:	e9 58 e1 ff ff       	jmpq   401190 <_ZN2cv8fastFreeEPv@plt>
  403038:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40303f:	00 
  403040:	5b                   	pop    %rbx
  403041:	c3                   	retq   
  403042:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403048:	e8 73 e0 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  40304d:	0f 1f 00             	nopl   (%rax)
  403050:	eb a1                	jmp    402ff3 <_ZN2cv3MatD1Ev+0x13>
  403052:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403059:	00 00 00 
  40305c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000403060 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev>:
  403060:	41 54                	push   %r12
  403062:	49 89 fc             	mov    %rdi,%r12
  403065:	55                   	push   %rbp
  403066:	53                   	push   %rbx
  403067:	48 8b 6f 08          	mov    0x8(%rdi),%rbp
  40306b:	48 8b 1f             	mov    (%rdi),%rbx
  40306e:	48 39 dd             	cmp    %rbx,%rbp
  403071:	75 5a                	jne    4030cd <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x6d>
  403073:	e9 95 00 00 00       	jmpq   40310d <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xad>
  403078:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40307f:	00 
  403080:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403084:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  40308b:	00 
  40308c:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403093:	00 
  403094:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40309b:	00 
  40309c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  4030a3:	00 
  4030a4:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4030aa:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  4030ae:	48 8d 43 50          	lea    0x50(%rbx),%rax
  4030b2:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  4030b9:	00 
  4030ba:	48 39 c7             	cmp    %rax,%rdi
  4030bd:	74 05                	je     4030c4 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x64>
  4030bf:	e8 cc e0 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  4030c4:	48 83 c3 60          	add    $0x60,%rbx
  4030c8:	48 39 dd             	cmp    %rbx,%rbp
  4030cb:	74 23                	je     4030f0 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x90>
  4030cd:	48 8b 43 18          	mov    0x18(%rbx),%rax
  4030d1:	48 85 c0             	test   %rax,%rax
  4030d4:	74 aa                	je     403080 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4030d6:	f0 83 28 01          	lock subl $0x1,(%rax)
  4030da:	75 a4                	jne    403080 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4030dc:	48 89 df             	mov    %rbx,%rdi
  4030df:	e8 dc df ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  4030e4:	eb 9a                	jmp    403080 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x20>
  4030e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4030ed:	00 00 00 
  4030f0:	49 8b 3c 24          	mov    (%r12),%rdi
  4030f4:	48 85 ff             	test   %rdi,%rdi
  4030f7:	74 0f                	je     403108 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xa8>
  4030f9:	5b                   	pop    %rbx
  4030fa:	5d                   	pop    %rbp
  4030fb:	41 5c                	pop    %r12
  4030fd:	e9 ce df ff ff       	jmpq   4010d0 <_ZdlPv@plt>
  403102:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403108:	5b                   	pop    %rbx
  403109:	5d                   	pop    %rbp
  40310a:	41 5c                	pop    %r12
  40310c:	c3                   	retq   
  40310d:	48 89 ef             	mov    %rbp,%rdi
  403110:	eb e2                	jmp    4030f4 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0x94>
  403112:	49 8b 3c 24          	mov    (%r12),%rdi
  403116:	48 89 c3             	mov    %rax,%rbx
  403119:	48 85 ff             	test   %rdi,%rdi
  40311c:	74 10                	je     40312e <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xce>
  40311e:	c5 f8 77             	vzeroupper 
  403121:	e8 aa df ff ff       	callq  4010d0 <_ZdlPv@plt>
  403126:	48 89 df             	mov    %rbx,%rdi
  403129:	e8 d2 df ff ff       	callq  401100 <_Unwind_Resume@plt>
  40312e:	c5 f8 77             	vzeroupper 
  403131:	eb f3                	jmp    403126 <_ZNSt6vectorIN2cv3MatESaIS1_EED1Ev+0xc6>
  403133:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40313a:	00 00 00 
  40313d:	0f 1f 00             	nopl   (%rax)

0000000000403140 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_>:
  403140:	55                   	push   %rbp
  403141:	48 89 e5             	mov    %rsp,%rbp
  403144:	41 57                	push   %r15
  403146:	49 89 ff             	mov    %rdi,%r15
  403149:	41 56                	push   %r14
  40314b:	49 89 f6             	mov    %rsi,%r14
  40314e:	41 55                	push   %r13
  403150:	41 54                	push   %r12
  403152:	53                   	push   %rbx
  403153:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  403157:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  40315e:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  403162:	49 3b 7f 10          	cmp    0x10(%r15),%rdi
  403166:	0f 84 3b 04 00 00    	je     4035a7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x467>
  40316c:	31 c0                	xor    %eax,%eax
  40316e:	48 85 ff             	test   %rdi,%rdi
  403171:	0f 84 8c 00 00 00    	je     403203 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  403177:	48 8b 4f c0          	mov    -0x40(%rdi),%rcx
  40317b:	48 8b 47 b0          	mov    -0x50(%rdi),%rax
  40317f:	c5 fa 6f 47 a0       	vmovdqu -0x60(%rdi),%xmm0
  403184:	48 89 4f 20          	mov    %rcx,0x20(%rdi)
  403188:	48 8b 4f c8          	mov    -0x38(%rdi),%rcx
  40318c:	48 89 47 10          	mov    %rax,0x10(%rdi)
  403190:	48 8b 47 b8          	mov    -0x48(%rdi),%rax
  403194:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  403198:	48 89 4f 28          	mov    %rcx,0x28(%rdi)
  40319c:	48 8b 4f d0          	mov    -0x30(%rdi),%rcx
  4031a0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4031a4:	48 85 c0             	test   %rax,%rax
  4031a7:	48 89 4f 30          	mov    %rcx,0x30(%rdi)
  4031ab:	48 8b 4f d8          	mov    -0x28(%rdi),%rcx
  4031af:	48 89 47 18          	mov    %rax,0x18(%rdi)
  4031b3:	c5 fa 7f 47 50       	vmovdqu %xmm0,0x50(%rdi)
  4031b8:	48 89 4f 38          	mov    %rcx,0x38(%rdi)
  4031bc:	48 8d 4f 08          	lea    0x8(%rdi),%rcx
  4031c0:	48 89 4f 40          	mov    %rcx,0x40(%rdi)
  4031c4:	48 8d 4f 50          	lea    0x50(%rdi),%rcx
  4031c8:	48 89 4f 48          	mov    %rcx,0x48(%rdi)
  4031cc:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  4031cf:	74 07                	je     4031d8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x98>
  4031d1:	f0 83 00 01          	lock addl $0x1,(%rax)
  4031d5:	8b 4f a4             	mov    -0x5c(%rdi),%ecx
  4031d8:	83 f9 02             	cmp    $0x2,%ecx
  4031db:	0f 8f 57 03 00 00    	jg     403538 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3f8>
  4031e1:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  4031e5:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4031e9:	48 8b 09             	mov    (%rcx),%rcx
  4031ec:	48 89 08             	mov    %rcx,(%rax)
  4031ef:	48 8b 4f e8          	mov    -0x18(%rdi),%rcx
  4031f3:	48 8b 47 48          	mov    0x48(%rdi),%rax
  4031f7:	48 8b 49 08          	mov    0x8(%rcx),%rcx
  4031fb:	48 89 48 08          	mov    %rcx,0x8(%rax)
  4031ff:	49 8b 47 08          	mov    0x8(%r15),%rax
  403203:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403207:	48 83 c0 60          	add    $0x60,%rax
  40320b:	4c 8d 6c 24 30       	lea    0x30(%rsp),%r13
  403210:	49 89 47 08          	mov    %rax,0x8(%r15)
  403214:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403218:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  40321c:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
  403221:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  403225:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40322a:	48 8b 42 18          	mov    0x18(%rdx),%rax
  40322e:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
  403234:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  403239:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  40323d:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403241:	48 85 c0             	test   %rax,%rax
  403244:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  403249:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  40324d:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  403252:	c5 f9 7f 84 24 80 00 	vmovdqa %xmm0,0x80(%rsp)
  403259:	00 00 
  40325b:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  403260:	49 8d 4d 08          	lea    0x8(%r13),%rcx
  403264:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
  403269:	49 8d 4d 50          	lea    0x50(%r13),%rcx
  40326d:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
  403272:	8b 4a 04             	mov    0x4(%rdx),%ecx
  403275:	74 07                	je     40327e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x13e>
  403277:	f0 83 00 01          	lock addl $0x1,(%rax)
  40327b:	8b 4a 04             	mov    0x4(%rdx),%ecx
  40327e:	83 f9 02             	cmp    $0x2,%ecx
  403281:	0f 8f d9 02 00 00    	jg     403560 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x420>
  403287:	48 8b 42 48          	mov    0x48(%rdx),%rax
  40328b:	48 8b 08             	mov    (%rax),%rcx
  40328e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  403293:	48 89 08             	mov    %rcx,(%rax)
  403296:	48 8b 42 48          	mov    0x48(%rdx),%rax
  40329a:	48 8b 50 08          	mov    0x8(%rax),%rdx
  40329e:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  4032a3:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4032a7:	4d 8b 67 08          	mov    0x8(%r15),%r12
  4032ab:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  4032b2:	aa aa aa 
  4032b5:	49 8d 5c 24 a0       	lea    -0x60(%r12),%rbx
  4032ba:	49 81 ec c0 00 00 00 	sub    $0xc0,%r12
  4032c1:	4d 29 f4             	sub    %r14,%r12
  4032c4:	49 c1 fc 05          	sar    $0x5,%r12
  4032c8:	4c 0f af e0          	imul   %rax,%r12
  4032cc:	4d 85 e4             	test   %r12,%r12
  4032cf:	7f 76                	jg     403347 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x207>
  4032d1:	e9 0a 01 00 00       	jmpq   4033e0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  4032d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4032dd:	00 00 00 
  4032e0:	89 43 04             	mov    %eax,0x4(%rbx)
  4032e3:	8b 43 a8             	mov    -0x58(%rbx),%eax
  4032e6:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  4032ea:	89 43 08             	mov    %eax,0x8(%rbx)
  4032ed:	8b 43 ac             	mov    -0x54(%rbx),%eax
  4032f0:	89 43 0c             	mov    %eax,0xc(%rbx)
  4032f3:	48 8b 43 48          	mov    0x48(%rbx),%rax
  4032f7:	48 8b 12             	mov    (%rdx),%rdx
  4032fa:	48 89 10             	mov    %rdx,(%rax)
  4032fd:	48 8b 53 e8          	mov    -0x18(%rbx),%rdx
  403301:	48 8b 43 48          	mov    0x48(%rbx),%rax
  403305:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403309:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40330d:	48 8b 43 b0          	mov    -0x50(%rbx),%rax
  403311:	48 89 43 10          	mov    %rax,0x10(%rbx)
  403315:	48 8b 43 c0          	mov    -0x40(%rbx),%rax
  403319:	48 89 43 20          	mov    %rax,0x20(%rbx)
  40331d:	48 8b 43 c8          	mov    -0x38(%rbx),%rax
  403321:	48 89 43 28          	mov    %rax,0x28(%rbx)
  403325:	48 8b 43 d0          	mov    -0x30(%rbx),%rax
  403329:	48 89 43 30          	mov    %rax,0x30(%rbx)
  40332d:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  403331:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403335:	48 8b 43 d8          	mov    -0x28(%rbx),%rax
  403339:	48 89 43 38          	mov    %rax,0x38(%rbx)
  40333d:	49 83 ec 01          	sub    $0x1,%r12
  403341:	0f 84 99 00 00 00    	je     4033e0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2a0>
  403347:	4c 8d bb 40 ff ff ff 	lea    -0xc0(%rbx),%r15
  40334e:	48 83 eb 60          	sub    $0x60,%rbx
  403352:	4c 39 fb             	cmp    %r15,%rbx
  403355:	74 e6                	je     40333d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1fd>
  403357:	48 8b 43 b8          	mov    -0x48(%rbx),%rax
  40335b:	48 85 c0             	test   %rax,%rax
  40335e:	74 04                	je     403364 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x224>
  403360:	f0 83 00 01          	lock addl $0x1,(%rax)
  403364:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403368:	48 85 c0             	test   %rax,%rax
  40336b:	74 06                	je     403373 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  40336d:	f0 83 28 01          	lock subl $0x1,(%rax)
  403371:	74 5d                	je     4033d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x290>
  403373:	48 8b 43 40          	mov    0x40(%rbx),%rax
  403377:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  40337e:	00 
  40337f:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403386:	00 
  403387:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40338e:	00 
  40338f:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403396:	00 
  403397:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40339d:	83 7b 04 02          	cmpl   $0x2,0x4(%rbx)
  4033a1:	8b 43 a0             	mov    -0x60(%rbx),%eax
  4033a4:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  4033ab:	00 
  4033ac:	89 03                	mov    %eax,(%rbx)
  4033ae:	7f 0c                	jg     4033bc <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x27c>
  4033b0:	8b 43 a4             	mov    -0x5c(%rbx),%eax
  4033b3:	83 f8 02             	cmp    $0x2,%eax
  4033b6:	0f 8e 24 ff ff ff    	jle    4032e0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1a0>
  4033bc:	4c 89 fe             	mov    %r15,%rsi
  4033bf:	48 89 df             	mov    %rbx,%rdi
  4033c2:	e8 99 dc ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4033c7:	e9 41 ff ff ff       	jmpq   40330d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x1cd>
  4033cc:	0f 1f 40 00          	nopl   0x0(%rax)
  4033d0:	48 89 df             	mov    %rbx,%rdi
  4033d3:	e8 e8 dc ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  4033d8:	eb 99                	jmp    403373 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x233>
  4033da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4033e0:	4d 39 ee             	cmp    %r13,%r14
  4033e3:	0f 84 b4 01 00 00    	je     40359d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x45d>
  4033e9:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4033ee:	48 85 c0             	test   %rax,%rax
  4033f1:	74 04                	je     4033f7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2b7>
  4033f3:	f0 83 00 01          	lock addl $0x1,(%rax)
  4033f7:	49 8b 46 18          	mov    0x18(%r14),%rax
  4033fb:	48 85 c0             	test   %rax,%rax
  4033fe:	74 0a                	je     40340a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  403400:	f0 83 28 01          	lock subl $0x1,(%rax)
  403404:	0f 84 86 01 00 00    	je     403590 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x450>
  40340a:	49 8b 46 40          	mov    0x40(%r14),%rax
  40340e:	49 c7 46 30 00 00 00 	movq   $0x0,0x30(%r14)
  403415:	00 
  403416:	49 c7 46 28 00 00 00 	movq   $0x0,0x28(%r14)
  40341d:	00 
  40341e:	49 c7 46 20 00 00 00 	movq   $0x0,0x20(%r14)
  403425:	00 
  403426:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  40342d:	00 
  40342e:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403434:	41 83 7e 04 02       	cmpl   $0x2,0x4(%r14)
  403439:	8b 44 24 30          	mov    0x30(%rsp),%eax
  40343d:	49 c7 46 18 00 00 00 	movq   $0x0,0x18(%r14)
  403444:	00 
  403445:	41 89 06             	mov    %eax,(%r14)
  403448:	7f 0d                	jg     403457 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x317>
  40344a:	8b 44 24 34          	mov    0x34(%rsp),%eax
  40344e:	83 f8 02             	cmp    $0x2,%eax
  403451:	0f 8e a1 00 00 00    	jle    4034f8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3b8>
  403457:	4c 89 ee             	mov    %r13,%rsi
  40345a:	4c 89 f7             	mov    %r14,%rdi
  40345d:	e8 fe db ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403462:	c5 f9 6f 44 24 40    	vmovdqa 0x40(%rsp),%xmm0
  403468:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40346d:	c4 c1 7a 7f 46 10    	vmovdqu %xmm0,0x10(%r14)
  403473:	c5 f9 6f 44 24 50    	vmovdqa 0x50(%rsp),%xmm0
  403479:	c4 c1 7a 7f 46 20    	vmovdqu %xmm0,0x20(%r14)
  40347f:	c5 f9 6f 44 24 60    	vmovdqa 0x60(%rsp),%xmm0
  403485:	c4 c1 7a 7f 46 30    	vmovdqu %xmm0,0x30(%r14)
  40348b:	48 85 c0             	test   %rax,%rax
  40348e:	74 0a                	je     40349a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  403490:	f0 83 28 01          	lock subl $0x1,(%rax)
  403494:	0f 84 e6 00 00 00    	je     403580 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x440>
  40349a:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
  40349f:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  4034a4:	49 83 c5 50          	add    $0x50,%r13
  4034a8:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  4034af:	00 00 
  4034b1:	48 c7 44 24 58 00 00 	movq   $0x0,0x58(%rsp)
  4034b8:	00 00 
  4034ba:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
  4034c1:	00 00 
  4034c3:	48 c7 44 24 40 00 00 	movq   $0x0,0x40(%rsp)
  4034ca:	00 00 
  4034cc:	4c 39 ef             	cmp    %r13,%rdi
  4034cf:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4034d5:	48 c7 44 24 48 00 00 	movq   $0x0,0x48(%rsp)
  4034dc:	00 00 
  4034de:	74 05                	je     4034e5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  4034e0:	e8 ab dc ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  4034e5:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  4034e9:	5b                   	pop    %rbx
  4034ea:	41 5c                	pop    %r12
  4034ec:	41 5d                	pop    %r13
  4034ee:	41 5e                	pop    %r14
  4034f0:	41 5f                	pop    %r15
  4034f2:	5d                   	pop    %rbp
  4034f3:	c3                   	retq   
  4034f4:	0f 1f 40 00          	nopl   0x0(%rax)
  4034f8:	41 89 46 04          	mov    %eax,0x4(%r14)
  4034fc:	8b 44 24 38          	mov    0x38(%rsp),%eax
  403500:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403505:	41 89 46 08          	mov    %eax,0x8(%r14)
  403509:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  40350d:	48 8b 12             	mov    (%rdx),%rdx
  403510:	41 89 46 0c          	mov    %eax,0xc(%r14)
  403514:	49 8b 46 48          	mov    0x48(%r14),%rax
  403518:	48 89 10             	mov    %rdx,(%rax)
  40351b:	48 8b 54 24 78       	mov    0x78(%rsp),%rdx
  403520:	49 8b 46 48          	mov    0x48(%r14),%rax
  403524:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  403528:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40352c:	e9 31 ff ff ff       	jmpq   403462 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x322>
  403531:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403538:	48 8d 77 a0          	lea    -0x60(%rdi),%rsi
  40353c:	c7 47 04 00 00 00 00 	movl   $0x0,0x4(%rdi)
  403543:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  403548:	e8 13 db ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  40354d:	49 8b 47 08          	mov    0x8(%r15),%rax
  403551:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  403556:	e9 a8 fc ff ff       	jmpq   403203 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0xc3>
  40355b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  403560:	48 89 d6             	mov    %rdx,%rsi
  403563:	4c 89 ef             	mov    %r13,%rdi
  403566:	c7 44 24 34 00 00 00 	movl   $0x0,0x34(%rsp)
  40356d:	00 
  40356e:	e8 ed da ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403573:	e9 2f fd ff ff       	jmpq   4032a7 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x167>
  403578:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40357f:	00 
  403580:	4c 89 ef             	mov    %r13,%rdi
  403583:	e8 38 db ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  403588:	e9 0d ff ff ff       	jmpq   40349a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x35a>
  40358d:	0f 1f 00             	nopl   (%rax)
  403590:	4c 89 f7             	mov    %r14,%rdi
  403593:	e8 28 db ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  403598:	e9 6d fe ff ff       	jmpq   40340a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x2ca>
  40359d:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4035a2:	e9 e4 fe ff ff       	jmpq   40348b <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x34b>
  4035a7:	49 8b 07             	mov    (%r15),%rax
  4035aa:	48 b9 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rcx
  4035b1:	aa aa aa 
  4035b4:	48 29 c7             	sub    %rax,%rdi
  4035b7:	48 c1 ff 05          	sar    $0x5,%rdi
  4035bb:	48 0f af f9          	imul   %rcx,%rdi
  4035bf:	48 85 ff             	test   %rdi,%rdi
  4035c2:	0f 85 be 01 00 00    	jne    403786 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x646>
  4035c8:	48 89 f3             	mov    %rsi,%rbx
  4035cb:	be 01 00 00 00       	mov    $0x1,%esi
  4035d0:	48 29 c3             	sub    %rax,%rbx
  4035d3:	48 c1 fb 05          	sar    $0x5,%rbx
  4035d7:	48 0f af d9          	imul   %rcx,%rbx
  4035db:	48 8d 3c 76          	lea    (%rsi,%rsi,2),%rdi
  4035df:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4035e4:	48 c1 e7 05          	shl    $0x5,%rdi
  4035e8:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  4035ed:	e8 8e da ff ff       	callq  401080 <_Znwm@plt>
  4035f2:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  4035f7:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4035fc:	48 8d 04 5b          	lea    (%rbx,%rbx,2),%rax
  403600:	48 c1 e0 05          	shl    $0x5,%rax
  403604:	48 03 44 24 18       	add    0x18(%rsp),%rax
  403609:	48 89 c6             	mov    %rax,%rsi
  40360c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  403611:	0f 84 87 00 00 00    	je     40369e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  403617:	c5 fa 6f 02          	vmovdqu (%rdx),%xmm0
  40361b:	c5 fa 7f 00          	vmovdqu %xmm0,(%rax)
  40361f:	48 8b 42 10          	mov    0x10(%rdx),%rax
  403623:	48 8b 4a 20          	mov    0x20(%rdx),%rcx
  403627:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  40362b:	48 89 46 10          	mov    %rax,0x10(%rsi)
  40362f:	48 8b 42 18          	mov    0x18(%rdx),%rax
  403633:	48 89 4e 20          	mov    %rcx,0x20(%rsi)
  403637:	48 8b 4a 28          	mov    0x28(%rdx),%rcx
  40363b:	48 85 c0             	test   %rax,%rax
  40363e:	48 89 46 18          	mov    %rax,0x18(%rsi)
  403642:	48 89 4e 28          	mov    %rcx,0x28(%rsi)
  403646:	48 8b 4a 30          	mov    0x30(%rdx),%rcx
  40364a:	48 89 4e 30          	mov    %rcx,0x30(%rsi)
  40364e:	48 8b 4a 38          	mov    0x38(%rdx),%rcx
  403652:	c5 fa 7f 46 50       	vmovdqu %xmm0,0x50(%rsi)
  403657:	48 89 4e 38          	mov    %rcx,0x38(%rsi)
  40365b:	48 8d 4e 08          	lea    0x8(%rsi),%rcx
  40365f:	48 89 4e 40          	mov    %rcx,0x40(%rsi)
  403663:	48 8d 4e 50          	lea    0x50(%rsi),%rcx
  403667:	48 89 4e 48          	mov    %rcx,0x48(%rsi)
  40366b:	74 04                	je     403671 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x531>
  40366d:	f0 83 00 01          	lock addl $0x1,(%rax)
  403671:	83 7a 04 02          	cmpl   $0x2,0x4(%rdx)
  403675:	0f 8f 0f 03 00 00    	jg     40398a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x84a>
  40367b:	48 8b 4a 48          	mov    0x48(%rdx),%rcx
  40367f:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  403684:	48 8b 09             	mov    (%rcx),%rcx
  403687:	48 8b 46 48          	mov    0x48(%rsi),%rax
  40368b:	48 89 08             	mov    %rcx,(%rax)
  40368e:	48 8b 52 48          	mov    0x48(%rdx),%rdx
  403692:	48 8b 46 48          	mov    0x48(%rsi),%rax
  403696:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40369a:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40369e:	4d 8b 27             	mov    (%r15),%r12
  4036a1:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  4036a6:	4d 39 e6             	cmp    %r12,%r14
  4036a9:	75 36                	jne    4036e1 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x5a1>
  4036ab:	e9 16 01 00 00       	jmpq   4037c6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  4036b0:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  4036b5:	48 8b 43 48          	mov    0x48(%rbx),%rax
  4036b9:	48 8b 12             	mov    (%rdx),%rdx
  4036bc:	48 89 10             	mov    %rdx,(%rax)
  4036bf:	49 8b 54 24 48       	mov    0x48(%r12),%rdx
  4036c4:	48 8b 43 48          	mov    0x48(%rbx),%rax
  4036c8:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  4036cc:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4036d0:	49 83 c4 60          	add    $0x60,%r12
  4036d4:	48 83 c3 60          	add    $0x60,%rbx
  4036d8:	4d 39 e6             	cmp    %r12,%r14
  4036db:	0f 84 e5 00 00 00    	je     4037c6 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x686>
  4036e1:	48 85 db             	test   %rbx,%rbx
  4036e4:	74 ea                	je     4036d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  4036e6:	41 8b 04 24          	mov    (%r12),%eax
  4036ea:	89 03                	mov    %eax,(%rbx)
  4036ec:	41 8b 44 24 04       	mov    0x4(%r12),%eax
  4036f1:	89 43 04             	mov    %eax,0x4(%rbx)
  4036f4:	41 8b 44 24 08       	mov    0x8(%r12),%eax
  4036f9:	89 43 08             	mov    %eax,0x8(%rbx)
  4036fc:	41 8b 44 24 0c       	mov    0xc(%r12),%eax
  403701:	89 43 0c             	mov    %eax,0xc(%rbx)
  403704:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  403709:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40370d:	49 8b 44 24 18       	mov    0x18(%r12),%rax
  403712:	48 89 43 18          	mov    %rax,0x18(%rbx)
  403716:	49 8b 54 24 20       	mov    0x20(%r12),%rdx
  40371b:	48 85 c0             	test   %rax,%rax
  40371e:	48 89 53 20          	mov    %rdx,0x20(%rbx)
  403722:	49 8b 54 24 28       	mov    0x28(%r12),%rdx
  403727:	48 89 53 28          	mov    %rdx,0x28(%rbx)
  40372b:	49 8b 54 24 30       	mov    0x30(%r12),%rdx
  403730:	48 89 53 30          	mov    %rdx,0x30(%rbx)
  403734:	49 8b 54 24 38       	mov    0x38(%r12),%rdx
  403739:	48 c7 43 58 00 00 00 	movq   $0x0,0x58(%rbx)
  403740:	00 
  403741:	48 c7 43 50 00 00 00 	movq   $0x0,0x50(%rbx)
  403748:	00 
  403749:	48 89 53 38          	mov    %rdx,0x38(%rbx)
  40374d:	48 8d 53 08          	lea    0x8(%rbx),%rdx
  403751:	48 89 53 40          	mov    %rdx,0x40(%rbx)
  403755:	48 8d 53 50          	lea    0x50(%rbx),%rdx
  403759:	48 89 53 48          	mov    %rdx,0x48(%rbx)
  40375d:	74 04                	je     403763 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x623>
  40375f:	f0 83 00 01          	lock addl $0x1,(%rax)
  403763:	41 83 7c 24 04 02    	cmpl   $0x2,0x4(%r12)
  403769:	0f 8e 41 ff ff ff    	jle    4036b0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x570>
  40376f:	c7 43 04 00 00 00 00 	movl   $0x0,0x4(%rbx)
  403776:	4c 89 e6             	mov    %r12,%rsi
  403779:	48 89 df             	mov    %rbx,%rdi
  40377c:	e8 df d8 ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  403781:	e9 4a ff ff ff       	jmpq   4036d0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x590>
  403786:	48 8d 34 3f          	lea    (%rdi,%rdi,1),%rsi
  40378a:	4c 89 f3             	mov    %r14,%rbx
  40378d:	48 29 c3             	sub    %rax,%rbx
  403790:	48 c1 fb 05          	sar    $0x5,%rbx
  403794:	48 39 f7             	cmp    %rsi,%rdi
  403797:	0f 86 79 02 00 00    	jbe    403a16 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d6>
  40379d:	48 b8 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%rax
  4037a4:	aa aa aa 
  4037a7:	48 c7 c7 c0 ff ff ff 	mov    $0xffffffffffffffc0,%rdi
  4037ae:	48 0f af d8          	imul   %rax,%rbx
  4037b2:	48 b8 aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rax
  4037b9:	aa aa 02 
  4037bc:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  4037c1:	e9 22 fe ff ff       	jmpq   4035e8 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4a8>
  4037c6:	49 8b 77 08          	mov    0x8(%r15),%rsi
  4037ca:	48 8d 43 60          	lea    0x60(%rbx),%rax
  4037ce:	4d 89 f5             	mov    %r14,%r13
  4037d1:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  4037d6:	49 89 c4             	mov    %rax,%r12
  4037d9:	49 39 f6             	cmp    %rsi,%r14
  4037dc:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  4037e1:	75 40                	jne    403823 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6e3>
  4037e3:	e9 fb 00 00 00       	jmpq   4038e3 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7a3>
  4037e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4037ef:	00 
  4037f0:	49 8b 55 48          	mov    0x48(%r13),%rdx
  4037f4:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  4037f9:	48 8b 12             	mov    (%rdx),%rdx
  4037fc:	48 89 10             	mov    %rdx,(%rax)
  4037ff:	49 8b 55 48          	mov    0x48(%r13),%rdx
  403803:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  403808:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40380c:	48 89 50 08          	mov    %rdx,0x8(%rax)
  403810:	49 83 c4 60          	add    $0x60,%r12
  403814:	49 83 c5 60          	add    $0x60,%r13
  403818:	4c 39 6c 24 20       	cmp    %r13,0x20(%rsp)
  40381d:	0f 84 bc 00 00 00    	je     4038df <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x79f>
  403823:	4d 85 e4             	test   %r12,%r12
  403826:	74 e8                	je     403810 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  403828:	41 8b 45 00          	mov    0x0(%r13),%eax
  40382c:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  403831:	49 8d 4c 24 08       	lea    0x8(%r12),%rcx
  403836:	41 89 04 24          	mov    %eax,(%r12)
  40383a:	41 8b 45 04          	mov    0x4(%r13),%eax
  40383e:	41 89 44 24 04       	mov    %eax,0x4(%r12)
  403843:	41 8b 45 08          	mov    0x8(%r13),%eax
  403847:	41 89 44 24 08       	mov    %eax,0x8(%r12)
  40384c:	41 8b 45 0c          	mov    0xc(%r13),%eax
  403850:	41 89 44 24 0c       	mov    %eax,0xc(%r12)
  403855:	49 8b 45 10          	mov    0x10(%r13),%rax
  403859:	49 89 44 24 10       	mov    %rax,0x10(%r12)
  40385e:	49 8b 45 18          	mov    0x18(%r13),%rax
  403862:	49 89 44 24 18       	mov    %rax,0x18(%r12)
  403867:	49 8b 55 20          	mov    0x20(%r13),%rdx
  40386b:	49 89 54 24 20       	mov    %rdx,0x20(%r12)
  403870:	49 8b 55 28          	mov    0x28(%r13),%rdx
  403874:	49 89 54 24 28       	mov    %rdx,0x28(%r12)
  403879:	49 8b 55 30          	mov    0x30(%r13),%rdx
  40387d:	49 89 54 24 30       	mov    %rdx,0x30(%r12)
  403882:	49 8b 55 38          	mov    0x38(%r13),%rdx
  403886:	49 89 54 24 38       	mov    %rdx,0x38(%r12)
  40388b:	4c 89 e2             	mov    %r12,%rdx
  40388e:	48 29 da             	sub    %rbx,%rdx
  403891:	48 85 c0             	test   %rax,%rax
  403894:	48 89 4c 16 e0       	mov    %rcx,-0x20(%rsi,%rdx,1)
  403899:	49 8d 4c 24 50       	lea    0x50(%r12),%rcx
  40389e:	48 89 4c 16 e8       	mov    %rcx,-0x18(%rsi,%rdx,1)
  4038a3:	49 c7 44 24 58 00 00 	movq   $0x0,0x58(%r12)
  4038aa:	00 00 
  4038ac:	49 c7 44 24 50 00 00 	movq   $0x0,0x50(%r12)
  4038b3:	00 00 
  4038b5:	74 04                	je     4038bb <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x77b>
  4038b7:	f0 83 00 01          	lock addl $0x1,(%rax)
  4038bb:	41 83 7d 04 02       	cmpl   $0x2,0x4(%r13)
  4038c0:	0f 8e 2a ff ff ff    	jle    4037f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6b0>
  4038c6:	41 c7 44 24 04 00 00 	movl   $0x0,0x4(%r12)
  4038cd:	00 00 
  4038cf:	4c 89 ee             	mov    %r13,%rsi
  4038d2:	4c 89 e7             	mov    %r12,%rdi
  4038d5:	e8 86 d7 ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4038da:	e9 31 ff ff ff       	jmpq   403810 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x6d0>
  4038df:	4d 8b 77 08          	mov    0x8(%r15),%r14
  4038e3:	49 8b 1f             	mov    (%r15),%rbx
  4038e6:	4c 89 f7             	mov    %r14,%rdi
  4038e9:	4c 39 f3             	cmp    %r14,%rbx
  4038ec:	75 4f                	jne    40393d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7fd>
  4038ee:	eb 69                	jmp    403959 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x819>
  4038f0:	48 8b 43 40          	mov    0x40(%rbx),%rax
  4038f4:	48 c7 43 30 00 00 00 	movq   $0x0,0x30(%rbx)
  4038fb:	00 
  4038fc:	48 c7 43 28 00 00 00 	movq   $0x0,0x28(%rbx)
  403903:	00 
  403904:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40390b:	00 
  40390c:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  403913:	00 
  403914:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40391a:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40391e:	48 8d 43 50          	lea    0x50(%rbx),%rax
  403922:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  403929:	00 
  40392a:	48 39 c7             	cmp    %rax,%rdi
  40392d:	74 05                	je     403934 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7f4>
  40392f:	e8 5c d8 ff ff       	callq  401190 <_ZN2cv8fastFreeEPv@plt>
  403934:	48 83 c3 60          	add    $0x60,%rbx
  403938:	4c 39 f3             	cmp    %r14,%rbx
  40393b:	74 19                	je     403956 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x816>
  40393d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  403941:	48 85 c0             	test   %rax,%rax
  403944:	74 aa                	je     4038f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403946:	f0 83 28 01          	lock subl $0x1,(%rax)
  40394a:	75 a4                	jne    4038f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  40394c:	48 89 df             	mov    %rbx,%rdi
  40394f:	e8 6c d7 ff ff       	callq  4010c0 <_ZN2cv3Mat10deallocateEv@plt>
  403954:	eb 9a                	jmp    4038f0 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x7b0>
  403956:	49 8b 3f             	mov    (%r15),%rdi
  403959:	48 85 ff             	test   %rdi,%rdi
  40395c:	74 07                	je     403965 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x825>
  40395e:	66 90                	xchg   %ax,%ax
  403960:	e8 6b d7 ff ff       	callq  4010d0 <_ZdlPv@plt>
  403965:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40396a:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  40396f:	4d 89 67 08          	mov    %r12,0x8(%r15)
  403973:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  403977:	49 89 37             	mov    %rsi,(%r15)
  40397a:	48 c1 e0 05          	shl    $0x5,%rax
  40397e:	48 01 f0             	add    %rsi,%rax
  403981:	49 89 47 10          	mov    %rax,0x10(%r15)
  403985:	e9 5b fb ff ff       	jmpq   4034e5 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x3a5>
  40398a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40398f:	48 89 d6             	mov    %rdx,%rsi
  403992:	c7 40 04 00 00 00 00 	movl   $0x0,0x4(%rax)
  403999:	48 89 c7             	mov    %rax,%rdi
  40399c:	e8 bf d6 ff ff       	callq  401060 <_ZN2cv3Mat8copySizeERKS0_@plt>
  4039a1:	e9 f8 fc ff ff       	jmpq   40369e <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x55e>
  4039a6:	48 89 c7             	mov    %rax,%rdi
  4039a9:	c5 f8 77             	vzeroupper 
  4039ac:	e8 8f d6 ff ff       	callq  401040 <__cxa_begin_catch@plt>
  4039b1:	4c 39 64 24 28       	cmp    %r12,0x28(%rsp)
  4039b6:	74 16                	je     4039ce <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x88e>
  4039b8:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
  4039bd:	48 89 df             	mov    %rbx,%rdi
  4039c0:	e8 1b f6 ff ff       	callq  402fe0 <_ZN2cv3MatD1Ev>
  4039c5:	48 83 c3 60          	add    $0x60,%rbx
  4039c9:	4c 39 e3             	cmp    %r12,%rbx
  4039cc:	75 ef                	jne    4039bd <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x87d>
  4039ce:	e8 8d d7 ff ff       	callq  401160 <__cxa_rethrow@plt>
  4039d3:	48 89 c3             	mov    %rax,%rbx
  4039d6:	4c 89 ef             	mov    %r13,%rdi
  4039d9:	c5 f8 77             	vzeroupper 
  4039dc:	e8 ff f5 ff ff       	callq  402fe0 <_ZN2cv3MatD1Ev>
  4039e1:	48 89 df             	mov    %rbx,%rdi
  4039e4:	e8 17 d7 ff ff       	callq  401100 <_Unwind_Resume@plt>
  4039e9:	48 89 c7             	mov    %rax,%rdi
  4039ec:	c5 f8 77             	vzeroupper 
  4039ef:	e8 4c d6 ff ff       	callq  401040 <__cxa_begin_catch@plt>
  4039f4:	48 39 5c 24 18       	cmp    %rbx,0x18(%rsp)
  4039f9:	74 16                	je     403a11 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8d1>
  4039fb:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  403a00:	4c 89 e7             	mov    %r12,%rdi
  403a03:	e8 d8 f5 ff ff       	callq  402fe0 <_ZN2cv3MatD1Ev>
  403a08:	49 83 c4 60          	add    $0x60,%r12
  403a0c:	4c 39 e3             	cmp    %r12,%rbx
  403a0f:	75 ef                	jne    403a00 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x8c0>
  403a11:	e8 4a d7 ff ff       	callq  401160 <__cxa_rethrow@plt>
  403a16:	48 bf aa aa aa aa aa 	movabs $0x2aaaaaaaaaaaaaa,%rdi
  403a1d:	aa aa 02 
  403a20:	48 39 fe             	cmp    %rdi,%rsi
  403a23:	0f 87 74 fd ff ff    	ja     40379d <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x65d>
  403a29:	48 0f af d9          	imul   %rcx,%rbx
  403a2d:	48 85 f6             	test   %rsi,%rsi
  403a30:	0f 85 a5 fb ff ff    	jne    4035db <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x49b>
  403a36:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  403a3d:	00 00 
  403a3f:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  403a46:	00 00 
  403a48:	e9 af fb ff ff       	jmpq   4035fc <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x4bc>
  403a4d:	48 89 c7             	mov    %rax,%rdi
  403a50:	c5 f8 77             	vzeroupper 
  403a53:	e8 e8 d5 ff ff       	callq  401040 <__cxa_begin_catch@plt>
  403a58:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  403a5e:	74 69                	je     403ac9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  403a60:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  403a65:	e8 66 d6 ff ff       	callq  4010d0 <_ZdlPv@plt>
  403a6a:	e8 f1 d6 ff ff       	callq  401160 <__cxa_rethrow@plt>
  403a6f:	48 89 c3             	mov    %rax,%rbx
  403a72:	c5 f8 77             	vzeroupper 
  403a75:	e8 c6 d6 ff ff       	callq  401140 <__cxa_end_catch@plt>
  403a7a:	48 89 df             	mov    %rbx,%rdi
  403a7d:	e8 be d5 ff ff       	callq  401040 <__cxa_begin_catch@plt>
  403a82:	48 83 7c 24 28 00    	cmpq   $0x0,0x28(%rsp)
  403a88:	74 3f                	je     403ac9 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x989>
  403a8a:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  403a8f:	48 39 44 24 28       	cmp    %rax,0x28(%rsp)
  403a94:	74 ca                	je     403a60 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  403a96:	48 89 c3             	mov    %rax,%rbx
  403a99:	48 89 df             	mov    %rbx,%rdi
  403a9c:	e8 3f f5 ff ff       	callq  402fe0 <_ZN2cv3MatD1Ev>
  403aa1:	48 83 c3 60          	add    $0x60,%rbx
  403aa5:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
  403aaa:	75 ed                	jne    403a99 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x959>
  403aac:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  403ab2:	75 ac                	jne    403a60 <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x920>
  403ab4:	eb b4                	jmp    403a6a <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x92a>
  403ab6:	48 89 c3             	mov    %rax,%rbx
  403ab9:	c5 f8 77             	vzeroupper 
  403abc:	e8 7f d6 ff ff       	callq  401140 <__cxa_end_catch@plt>
  403ac1:	48 89 df             	mov    %rbx,%rdi
  403ac4:	e8 77 d5 ff ff       	callq  401040 <__cxa_begin_catch@plt>
  403ac9:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  403ace:	e8 0d f5 ff ff       	callq  402fe0 <_ZN2cv3MatD1Ev>
  403ad3:	eb d7                	jmp    403aac <_ZNSt6vectorIN2cv3MatESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_+0x96c>
  403ad5:	48 89 c3             	mov    %rax,%rbx
  403ad8:	c5 f8 77             	vzeroupper 
  403adb:	e8 60 d6 ff ff       	callq  401140 <__cxa_end_catch@plt>
  403ae0:	48 89 df             	mov    %rbx,%rdi
  403ae3:	e8 18 d6 ff ff       	callq  401100 <_Unwind_Resume@plt>
  403ae8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403aef:	00 

0000000000403af0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag>:
  403af0:	55                   	push   %rbp
  403af1:	48 89 e5             	mov    %rsp,%rbp
  403af4:	41 56                	push   %r14
  403af6:	41 55                	push   %r13
  403af8:	41 54                	push   %r12
  403afa:	49 89 d4             	mov    %rdx,%r12
  403afd:	49 29 f4             	sub    %rsi,%r12
  403b00:	53                   	push   %rbx
  403b01:	48 89 fb             	mov    %rdi,%rbx
  403b04:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  403b08:	48 83 ec 10          	sub    $0x10,%rsp
  403b0c:	48 8b 07             	mov    (%rdi),%rax
  403b0f:	48 8b 4f 10          	mov    0x10(%rdi),%rcx
  403b13:	48 29 c1             	sub    %rax,%rcx
  403b16:	48 c1 f9 02          	sar    $0x2,%rcx
  403b1a:	49 39 cc             	cmp    %rcx,%r12
  403b1d:	0f 87 0d 04 00 00    	ja     403f30 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x440>
  403b23:	4c 8b 4f 08          	mov    0x8(%rdi),%r9
  403b27:	4c 89 cf             	mov    %r9,%rdi
  403b2a:	48 29 c7             	sub    %rax,%rdi
  403b2d:	48 c1 ff 02          	sar    $0x2,%rdi
  403b31:	49 39 fc             	cmp    %rdi,%r12
  403b34:	0f 87 2e 01 00 00    	ja     403c68 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x178>
  403b3a:	4d 85 e4             	test   %r12,%r12
  403b3d:	0f 8e a5 05 00 00    	jle    4040e8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5f8>
  403b43:	4a 8d 3c a0          	lea    (%rax,%r12,4),%rdi
  403b47:	4a 8d 0c 26          	lea    (%rsi,%r12,1),%rcx
  403b4b:	48 39 fe             	cmp    %rdi,%rsi
  403b4e:	41 0f 93 c0          	setae  %r8b
  403b52:	48 39 c8             	cmp    %rcx,%rax
  403b55:	0f 93 c2             	setae  %dl
  403b58:	41 08 d0             	or     %dl,%r8b
  403b5b:	0f 84 8f 03 00 00    	je     403ef0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403b61:	49 83 fc 1f          	cmp    $0x1f,%r12
  403b65:	0f 86 85 03 00 00    	jbe    403ef0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403b6b:	4d 89 e0             	mov    %r12,%r8
  403b6e:	48 8d 50 20          	lea    0x20(%rax),%rdx
  403b72:	45 31 c9             	xor    %r9d,%r9d
  403b75:	49 c1 e8 05          	shr    $0x5,%r8
  403b79:	4c 89 c1             	mov    %r8,%rcx
  403b7c:	48 c1 e1 05          	shl    $0x5,%rcx
  403b80:	4d 89 ca             	mov    %r9,%r10
  403b83:	49 83 c1 01          	add    $0x1,%r9
  403b87:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  403b8b:	49 c1 e2 05          	shl    $0x5,%r10
  403b8f:	c4 a1 7a 6f 04 16    	vmovdqu (%rsi,%r10,1),%xmm0
  403b95:	c4 a3 7d 38 44 16 10 	vinserti128 $0x1,0x10(%rsi,%r10,1),%ymm0,%ymm0
  403b9c:	01 
  403b9d:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403ba2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403ba8:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403bad:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403bb3:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403bb8:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403bbc:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  403bc3:	ff 
  403bc4:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403bc9:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  403bd0:	ff ff 01 
  403bd3:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403bd7:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  403bdc:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  403be3:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403be8:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403bee:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403bf2:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  403bf7:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  403bfe:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403c03:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403c07:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  403c0c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  403c13:	4d 39 c1             	cmp    %r8,%r9
  403c16:	0f 82 64 ff ff ff    	jb     403b80 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x90>
  403c1c:	48 8d 14 88          	lea    (%rax,%rcx,4),%rdx
  403c20:	4c 89 e0             	mov    %r12,%rax
  403c23:	48 01 ce             	add    %rcx,%rsi
  403c26:	48 29 c8             	sub    %rcx,%rax
  403c29:	49 39 cc             	cmp    %rcx,%r12
  403c2c:	74 1f                	je     403c4d <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x15d>
  403c2e:	66 90                	xchg   %ax,%ax
  403c30:	0f b6 0e             	movzbl (%rsi),%ecx
  403c33:	48 83 e8 01          	sub    $0x1,%rax
  403c37:	48 83 c6 01          	add    $0x1,%rsi
  403c3b:	48 83 c2 04          	add    $0x4,%rdx
  403c3f:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  403c43:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  403c48:	48 85 c0             	test   %rax,%rax
  403c4b:	7f e3                	jg     403c30 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x140>
  403c4d:	c5 f8 77             	vzeroupper 
  403c50:	48 89 7b 08          	mov    %rdi,0x8(%rbx)
  403c54:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  403c58:	5b                   	pop    %rbx
  403c59:	41 5c                	pop    %r12
  403c5b:	41 5d                	pop    %r13
  403c5d:	41 5e                	pop    %r14
  403c5f:	5d                   	pop    %rbp
  403c60:	c3                   	retq   
  403c61:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403c68:	48 01 f7             	add    %rsi,%rdi
  403c6b:	49 89 fa             	mov    %rdi,%r10
  403c6e:	49 29 f2             	sub    %rsi,%r10
  403c71:	4d 85 d2             	test   %r10,%r10
  403c74:	0f 8e 14 01 00 00    	jle    403d8e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403c7a:	4a 8d 0c 90          	lea    (%rax,%r10,4),%rcx
  403c7e:	4e 8d 04 16          	lea    (%rsi,%r10,1),%r8
  403c82:	48 39 ce             	cmp    %rcx,%rsi
  403c85:	41 0f 93 c3          	setae  %r11b
  403c89:	4c 39 c0             	cmp    %r8,%rax
  403c8c:	0f 93 c1             	setae  %cl
  403c8f:	41 08 cb             	or     %cl,%r11b
  403c92:	0f 84 30 02 00 00    	je     403ec8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403c98:	49 83 fa 1f          	cmp    $0x1f,%r10
  403c9c:	0f 86 26 02 00 00    	jbe    403ec8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403ca2:	4d 89 d0             	mov    %r10,%r8
  403ca5:	48 8d 48 20          	lea    0x20(%rax),%rcx
  403ca9:	45 31 e4             	xor    %r12d,%r12d
  403cac:	49 c1 e8 05          	shr    $0x5,%r8
  403cb0:	4d 89 c3             	mov    %r8,%r11
  403cb3:	49 c1 e3 05          	shl    $0x5,%r11
  403cb7:	4d 89 e5             	mov    %r12,%r13
  403cba:	49 83 c4 01          	add    $0x1,%r12
  403cbe:	48 83 e9 80          	sub    $0xffffffffffffff80,%rcx
  403cc2:	49 c1 e5 05          	shl    $0x5,%r13
  403cc6:	c4 a1 7a 6f 04 2e    	vmovdqu (%rsi,%r13,1),%xmm0
  403ccc:	c4 a3 7d 38 44 2e 10 	vinserti128 $0x1,0x10(%rsi,%r13,1),%ymm0,%ymm0
  403cd3:	01 
  403cd4:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403cd9:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403cdf:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403ce4:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403cea:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403cef:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403cf3:	c5 f8 11 91 60 ff ff 	vmovups %xmm2,-0xa0(%rcx)
  403cfa:	ff 
  403cfb:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403d00:	c4 e3 7d 19 91 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rcx)
  403d07:	ff ff 01 
  403d0a:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403d0e:	c5 f8 11 49 80       	vmovups %xmm1,-0x80(%rcx)
  403d13:	c4 e3 7d 19 49 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rcx)
  403d1a:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403d1f:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403d25:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403d29:	c5 f8 11 49 a0       	vmovups %xmm1,-0x60(%rcx)
  403d2e:	c4 e3 7d 19 49 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rcx)
  403d35:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403d3a:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403d3e:	c5 f8 11 41 c0       	vmovups %xmm0,-0x40(%rcx)
  403d43:	c4 e3 7d 19 41 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rcx)
  403d4a:	4d 39 c4             	cmp    %r8,%r12
  403d4d:	0f 82 64 ff ff ff    	jb     403cb7 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x1c7>
  403d53:	4a 8d 0c 98          	lea    (%rax,%r11,4),%rcx
  403d57:	4c 89 d0             	mov    %r10,%rax
  403d5a:	4e 8d 04 1e          	lea    (%rsi,%r11,1),%r8
  403d5e:	4c 29 d8             	sub    %r11,%rax
  403d61:	4d 39 da             	cmp    %r11,%r10
  403d64:	74 28                	je     403d8e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403d66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403d6d:	00 00 00 
  403d70:	41 0f b6 30          	movzbl (%r8),%esi
  403d74:	48 83 e8 01          	sub    $0x1,%rax
  403d78:	49 83 c0 01          	add    $0x1,%r8
  403d7c:	48 83 c1 04          	add    $0x4,%rcx
  403d80:	c5 fa 2a c6          	vcvtsi2ss %esi,%xmm0,%xmm0
  403d84:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  403d89:	48 85 c0             	test   %rax,%rax
  403d8c:	7f e2                	jg     403d70 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x280>
  403d8e:	48 29 fa             	sub    %rdi,%rdx
  403d91:	4d 89 c8             	mov    %r9,%r8
  403d94:	48 85 d2             	test   %rdx,%rdx
  403d97:	0f 8e 10 01 00 00    	jle    403ead <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  403d9d:	4d 8d 04 91          	lea    (%r9,%rdx,4),%r8
  403da1:	48 8d 0c 17          	lea    (%rdi,%rdx,1),%rcx
  403da5:	4c 39 c7             	cmp    %r8,%rdi
  403da8:	40 0f 93 c6          	setae  %sil
  403dac:	49 39 c9             	cmp    %rcx,%r9
  403daf:	0f 93 c0             	setae  %al
  403db2:	40 08 c6             	or     %al,%sil
  403db5:	0f 84 55 01 00 00    	je     403f10 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  403dbb:	48 83 fa 1f          	cmp    $0x1f,%rdx
  403dbf:	0f 86 4b 01 00 00    	jbe    403f10 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  403dc5:	48 89 d1             	mov    %rdx,%rcx
  403dc8:	49 8d 41 20          	lea    0x20(%r9),%rax
  403dcc:	31 f6                	xor    %esi,%esi
  403dce:	48 c1 e9 05          	shr    $0x5,%rcx
  403dd2:	49 89 ca             	mov    %rcx,%r10
  403dd5:	49 c1 e2 05          	shl    $0x5,%r10
  403dd9:	49 89 f3             	mov    %rsi,%r11
  403ddc:	48 83 c6 01          	add    $0x1,%rsi
  403de0:	48 83 e8 80          	sub    $0xffffffffffffff80,%rax
  403de4:	49 c1 e3 05          	shl    $0x5,%r11
  403de8:	c4 a1 7a 6f 04 1f    	vmovdqu (%rdi,%r11,1),%xmm0
  403dee:	c4 a3 7d 38 44 1f 10 	vinserti128 $0x1,0x10(%rdi,%r11,1),%ymm0,%ymm0
  403df5:	01 
  403df6:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403dfb:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403e01:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403e06:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403e0c:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403e11:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403e15:	c5 f8 11 90 60 ff ff 	vmovups %xmm2,-0xa0(%rax)
  403e1c:	ff 
  403e1d:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403e22:	c4 e3 7d 19 90 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rax)
  403e29:	ff ff 01 
  403e2c:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403e30:	c5 f8 11 48 80       	vmovups %xmm1,-0x80(%rax)
  403e35:	c4 e3 7d 19 48 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rax)
  403e3c:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  403e41:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403e47:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  403e4b:	c5 f8 11 48 a0       	vmovups %xmm1,-0x60(%rax)
  403e50:	c4 e3 7d 19 48 b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rax)
  403e57:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  403e5c:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  403e60:	c5 f8 11 40 c0       	vmovups %xmm0,-0x40(%rax)
  403e65:	c4 e3 7d 19 40 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rax)
  403e6c:	48 39 ce             	cmp    %rcx,%rsi
  403e6f:	0f 82 64 ff ff ff    	jb     403dd9 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x2e9>
  403e75:	48 89 d0             	mov    %rdx,%rax
  403e78:	4a 8d 34 17          	lea    (%rdi,%r10,1),%rsi
  403e7c:	4b 8d 0c 91          	lea    (%r9,%r10,4),%rcx
  403e80:	4c 29 d0             	sub    %r10,%rax
  403e83:	4c 39 d2             	cmp    %r10,%rdx
  403e86:	74 25                	je     403ead <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  403e88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403e8f:	00 
  403e90:	0f b6 16             	movzbl (%rsi),%edx
  403e93:	48 83 e8 01          	sub    $0x1,%rax
  403e97:	48 83 c6 01          	add    $0x1,%rsi
  403e9b:	48 83 c1 04          	add    $0x4,%rcx
  403e9f:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  403ea3:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  403ea8:	48 85 c0             	test   %rax,%rax
  403eab:	7f e3                	jg     403e90 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3a0>
  403ead:	4c 89 43 08          	mov    %r8,0x8(%rbx)
  403eb1:	c5 f8 77             	vzeroupper 
  403eb4:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  403eb8:	5b                   	pop    %rbx
  403eb9:	41 5c                	pop    %r12
  403ebb:	41 5d                	pop    %r13
  403ebd:	41 5e                	pop    %r14
  403ebf:	5d                   	pop    %rbp
  403ec0:	c3                   	retq   
  403ec1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403ec8:	0f b6 0e             	movzbl (%rsi),%ecx
  403ecb:	48 83 c6 01          	add    $0x1,%rsi
  403ecf:	48 83 c0 04          	add    $0x4,%rax
  403ed3:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  403ed7:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  403edc:	49 39 f0             	cmp    %rsi,%r8
  403edf:	75 e7                	jne    403ec8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3d8>
  403ee1:	e9 a8 fe ff ff       	jmpq   403d8e <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x29e>
  403ee6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403eed:	00 00 00 
  403ef0:	0f b6 16             	movzbl (%rsi),%edx
  403ef3:	48 83 c6 01          	add    $0x1,%rsi
  403ef7:	48 83 c0 04          	add    $0x4,%rax
  403efb:	c5 fa 2a c2          	vcvtsi2ss %edx,%xmm0,%xmm0
  403eff:	c5 fa 11 40 fc       	vmovss %xmm0,-0x4(%rax)
  403f04:	48 39 f1             	cmp    %rsi,%rcx
  403f07:	75 e7                	jne    403ef0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x400>
  403f09:	e9 42 fd ff ff       	jmpq   403c50 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  403f0e:	66 90                	xchg   %ax,%ax
  403f10:	0f b6 07             	movzbl (%rdi),%eax
  403f13:	48 83 c7 01          	add    $0x1,%rdi
  403f17:	49 83 c1 04          	add    $0x4,%r9
  403f1b:	c5 fa 2a c0          	vcvtsi2ss %eax,%xmm0,%xmm0
  403f1f:	c4 c1 7a 11 41 fc    	vmovss %xmm0,-0x4(%r9)
  403f25:	48 39 f9             	cmp    %rdi,%rcx
  403f28:	75 e6                	jne    403f10 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x420>
  403f2a:	eb 81                	jmp    403ead <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x3bd>
  403f2c:	0f 1f 40 00          	nopl   0x0(%rax)
  403f30:	4d 85 e4             	test   %r12,%r12
  403f33:	0f 84 9f 01 00 00    	je     4040d8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5e8>
  403f39:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rax
  403f40:	ff ff 3f 
  403f43:	49 39 c4             	cmp    %rax,%r12
  403f46:	0f 87 a4 01 00 00    	ja     4040f0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x600>
  403f4c:	4e 8d 34 a5 00 00 00 	lea    0x0(,%r12,4),%r14
  403f53:	00 
  403f54:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  403f59:	4c 89 f7             	mov    %r14,%rdi
  403f5c:	e8 1f d1 ff ff       	callq  401080 <_Znwm@plt>
  403f61:	4d 85 e4             	test   %r12,%r12
  403f64:	49 89 c5             	mov    %rax,%r13
  403f67:	48 8b 3b             	mov    (%rbx),%rdi
  403f6a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  403f6f:	0f 8e 0d 01 00 00    	jle    404082 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  403f75:	4a 8d 04 30          	lea    (%rax,%r14,1),%rax
  403f79:	4e 8d 04 26          	lea    (%rsi,%r12,1),%r8
  403f7d:	48 39 c6             	cmp    %rax,%rsi
  403f80:	0f 93 c2             	setae  %dl
  403f83:	4d 39 e8             	cmp    %r13,%r8
  403f86:	0f 96 c0             	setbe  %al
  403f89:	08 c2                	or     %al,%dl
  403f8b:	0f 84 1f 01 00 00    	je     4040b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  403f91:	49 83 fc 1f          	cmp    $0x1f,%r12
  403f95:	0f 86 15 01 00 00    	jbe    4040b0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c0>
  403f9b:	4c 89 e1             	mov    %r12,%rcx
  403f9e:	49 8d 55 20          	lea    0x20(%r13),%rdx
  403fa2:	45 31 c0             	xor    %r8d,%r8d
  403fa5:	48 c1 e9 05          	shr    $0x5,%rcx
  403fa9:	48 89 c8             	mov    %rcx,%rax
  403fac:	48 c1 e0 05          	shl    $0x5,%rax
  403fb0:	4d 89 c1             	mov    %r8,%r9
  403fb3:	49 83 c0 01          	add    $0x1,%r8
  403fb7:	48 83 ea 80          	sub    $0xffffffffffffff80,%rdx
  403fbb:	49 c1 e1 05          	shl    $0x5,%r9
  403fbf:	c4 a1 7a 6f 04 0e    	vmovdqu (%rsi,%r9,1),%xmm0
  403fc5:	c4 a3 7d 38 44 0e 10 	vinserti128 $0x1,0x10(%rsi,%r9,1),%ymm0,%ymm0
  403fcc:	01 
  403fcd:	c4 e2 7d 30 c8       	vpmovzxbw %xmm0,%ymm1
  403fd2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  403fd8:	c4 e2 7d 23 d1       	vpmovsxwd %xmm1,%ymm2
  403fdd:	c4 e3 7d 39 c9 01    	vextracti128 $0x1,%ymm1,%xmm1
  403fe3:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  403fe8:	c5 fc 5b d2          	vcvtdq2ps %ymm2,%ymm2
  403fec:	c5 f8 11 92 60 ff ff 	vmovups %xmm2,-0xa0(%rdx)
  403ff3:	ff 
  403ff4:	c4 e2 7d 23 c9       	vpmovsxwd %xmm1,%ymm1
  403ff9:	c4 e3 7d 19 92 70 ff 	vextractf128 $0x1,%ymm2,-0x90(%rdx)
  404000:	ff ff 01 
  404003:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404007:	c5 f8 11 4a 80       	vmovups %xmm1,-0x80(%rdx)
  40400c:	c4 e3 7d 19 4a 90 01 	vextractf128 $0x1,%ymm1,-0x70(%rdx)
  404013:	c4 e2 7d 23 c8       	vpmovsxwd %xmm0,%ymm1
  404018:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  40401e:	c5 fc 5b c9          	vcvtdq2ps %ymm1,%ymm1
  404022:	c5 f8 11 4a a0       	vmovups %xmm1,-0x60(%rdx)
  404027:	c4 e3 7d 19 4a b0 01 	vextractf128 $0x1,%ymm1,-0x50(%rdx)
  40402e:	c4 e2 7d 23 c0       	vpmovsxwd %xmm0,%ymm0
  404033:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  404037:	c5 f8 11 42 c0       	vmovups %xmm0,-0x40(%rdx)
  40403c:	c4 e3 7d 19 42 d0 01 	vextractf128 $0x1,%ymm0,-0x30(%rdx)
  404043:	4c 39 c1             	cmp    %r8,%rcx
  404046:	0f 87 64 ff ff ff    	ja     403fb0 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x4c0>
  40404c:	4c 89 e2             	mov    %r12,%rdx
  40404f:	48 01 c6             	add    %rax,%rsi
  404052:	49 8d 4c 85 00       	lea    0x0(%r13,%rax,4),%rcx
  404057:	48 29 c2             	sub    %rax,%rdx
  40405a:	49 39 c4             	cmp    %rax,%r12
  40405d:	74 20                	je     40407f <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x58f>
  40405f:	90                   	nop
  404060:	44 0f b6 06          	movzbl (%rsi),%r8d
  404064:	48 83 ea 01          	sub    $0x1,%rdx
  404068:	48 83 c6 01          	add    $0x1,%rsi
  40406c:	48 83 c1 04          	add    $0x4,%rcx
  404070:	c4 c1 7a 2a c0       	vcvtsi2ss %r8d,%xmm0,%xmm0
  404075:	c5 fa 11 41 fc       	vmovss %xmm0,-0x4(%rcx)
  40407a:	48 85 d2             	test   %rdx,%rdx
  40407d:	7f e1                	jg     404060 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x570>
  40407f:	c5 f8 77             	vzeroupper 
  404082:	48 85 ff             	test   %rdi,%rdi
  404085:	74 05                	je     40408c <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x59c>
  404087:	e8 44 d0 ff ff       	callq  4010d0 <_ZdlPv@plt>
  40408c:	4c 89 2b             	mov    %r13,(%rbx)
  40408f:	4d 01 f5             	add    %r14,%r13
  404092:	4c 89 6b 08          	mov    %r13,0x8(%rbx)
  404096:	4c 89 6b 10          	mov    %r13,0x10(%rbx)
  40409a:	48 8d 65 e0          	lea    -0x20(%rbp),%rsp
  40409e:	5b                   	pop    %rbx
  40409f:	41 5c                	pop    %r12
  4040a1:	41 5d                	pop    %r13
  4040a3:	41 5e                	pop    %r14
  4040a5:	5d                   	pop    %rbp
  4040a6:	c3                   	retq   
  4040a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4040ae:	00 00 
  4040b0:	4c 89 ea             	mov    %r13,%rdx
  4040b3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4040b8:	0f b6 0e             	movzbl (%rsi),%ecx
  4040bb:	48 83 c6 01          	add    $0x1,%rsi
  4040bf:	48 83 c2 04          	add    $0x4,%rdx
  4040c3:	c5 fa 2a c1          	vcvtsi2ss %ecx,%xmm0,%xmm0
  4040c7:	c5 fa 11 42 fc       	vmovss %xmm0,-0x4(%rdx)
  4040cc:	49 39 f0             	cmp    %rsi,%r8
  4040cf:	75 e7                	jne    4040b8 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x5c8>
  4040d1:	eb af                	jmp    404082 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4040d3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4040d8:	48 89 c7             	mov    %rax,%rdi
  4040db:	45 31 f6             	xor    %r14d,%r14d
  4040de:	45 31 ed             	xor    %r13d,%r13d
  4040e1:	eb 9f                	jmp    404082 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x592>
  4040e3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4040e8:	48 89 c7             	mov    %rax,%rdi
  4040eb:	e9 60 fb ff ff       	jmpq   403c50 <_ZNSt6vectorIfSaIfEE13_M_assign_auxIPhEEvT_S4_St20forward_iterator_tag+0x160>
  4040f0:	e8 7b cf ff ff       	callq  401070 <_ZSt17__throw_bad_allocv@plt>
  4040f5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4040fc:	00 00 00 
  4040ff:	90                   	nop

0000000000404100 <__libc_csu_init>:
  404100:	41 57                	push   %r15
  404102:	41 89 ff             	mov    %edi,%r15d
  404105:	41 56                	push   %r14
  404107:	49 89 f6             	mov    %rsi,%r14
  40410a:	41 55                	push   %r13
  40410c:	49 89 d5             	mov    %rdx,%r13
  40410f:	41 54                	push   %r12
  404111:	4c 8d 25 c0 0b 20 00 	lea    0x200bc0(%rip),%r12        # 604cd8 <__frame_dummy_init_array_entry>
  404118:	55                   	push   %rbp
  404119:	48 8d 2d c8 0b 20 00 	lea    0x200bc8(%rip),%rbp        # 604ce8 <__init_array_end>
  404120:	53                   	push   %rbx
  404121:	4c 29 e5             	sub    %r12,%rbp
  404124:	31 db                	xor    %ebx,%ebx
  404126:	48 c1 fd 03          	sar    $0x3,%rbp
  40412a:	48 83 ec 08          	sub    $0x8,%rsp
  40412e:	e8 d5 ce ff ff       	callq  401008 <_init>
  404133:	48 85 ed             	test   %rbp,%rbp
  404136:	74 1e                	je     404156 <__libc_csu_init+0x56>
  404138:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40413f:	00 
  404140:	4c 89 ea             	mov    %r13,%rdx
  404143:	4c 89 f6             	mov    %r14,%rsi
  404146:	44 89 ff             	mov    %r15d,%edi
  404149:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40414d:	48 83 c3 01          	add    $0x1,%rbx
  404151:	48 39 eb             	cmp    %rbp,%rbx
  404154:	75 ea                	jne    404140 <__libc_csu_init+0x40>
  404156:	48 83 c4 08          	add    $0x8,%rsp
  40415a:	5b                   	pop    %rbx
  40415b:	5d                   	pop    %rbp
  40415c:	41 5c                	pop    %r12
  40415e:	41 5d                	pop    %r13
  404160:	41 5e                	pop    %r14
  404162:	41 5f                	pop    %r15
  404164:	c3                   	retq   
  404165:	90                   	nop
  404166:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40416d:	00 00 00 

0000000000404170 <__libc_csu_fini>:
  404170:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000404174 <_fini>:
  404174:	48 83 ec 08          	sub    $0x8,%rsp
  404178:	48 83 c4 08          	add    $0x8,%rsp
  40417c:	c3                   	retq   
