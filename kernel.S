
./build/kernel:     file format elf64-x86-64


Disassembly of section .init:

0000000000400a90 <_init>:
  400a90:	48 83 ec 08          	sub    $0x8,%rsp
  400a94:	48 8b 05 5d 15 20 00 	mov    0x20155d(%rip),%rax        # 601ff8 <__gmon_start__>
  400a9b:	48 85 c0             	test   %rax,%rax
  400a9e:	74 05                	je     400aa5 <_init+0x15>
  400aa0:	e8 3b 00 00 00       	callq  400ae0 <__gmon_start__@plt>
  400aa5:	48 83 c4 08          	add    $0x8,%rsp
  400aa9:	c3                   	retq   

Disassembly of section .plt:

0000000000400ab0 <.plt>:
  400ab0:	ff 35 52 15 20 00    	pushq  0x201552(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400ab6:	ff 25 54 15 20 00    	jmpq   *0x201554(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400abc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ac0 <free@plt>:
  400ac0:	ff 25 52 15 20 00    	jmpq   *0x201552(%rip)        # 602018 <free@GLIBC_2.2.5>
  400ac6:	68 00 00 00 00       	pushq  $0x0
  400acb:	e9 e0 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400ad0 <_ZSt16__throw_bad_castv@plt>:
  400ad0:	ff 25 4a 15 20 00    	jmpq   *0x20154a(%rip)        # 602020 <_ZSt16__throw_bad_castv@GLIBCXX_3.4>
  400ad6:	68 01 00 00 00       	pushq  $0x1
  400adb:	e9 d0 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400ae0 <__gmon_start__@plt>:
  400ae0:	ff 25 42 15 20 00    	jmpq   *0x201542(%rip)        # 602028 <__gmon_start__>
  400ae6:	68 02 00 00 00       	pushq  $0x2
  400aeb:	e9 c0 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400af0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>:
  400af0:	ff 25 3a 15 20 00    	jmpq   *0x20153a(%rip)        # 602030 <_ZNKSt5ctypeIcE13_M_widen_initEv@GLIBCXX_3.4.11>
  400af6:	68 03 00 00 00       	pushq  $0x3
  400afb:	e9 b0 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b00 <_ZNSo5flushEv@plt>:
  400b00:	ff 25 32 15 20 00    	jmpq   *0x201532(%rip)        # 602038 <_ZNSo5flushEv@GLIBCXX_3.4>
  400b06:	68 04 00 00 00       	pushq  $0x4
  400b0b:	e9 a0 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b10 <_ZNSo3putEc@plt>:
  400b10:	ff 25 2a 15 20 00    	jmpq   *0x20152a(%rip)        # 602040 <_ZNSo3putEc@GLIBCXX_3.4>
  400b16:	68 05 00 00 00       	pushq  $0x5
  400b1b:	e9 90 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b20 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
  400b20:	ff 25 22 15 20 00    	jmpq   *0x201522(%rip)        # 602048 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
  400b26:	68 06 00 00 00       	pushq  $0x6
  400b2b:	e9 80 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b30 <_ZNSt8ios_base4InitD1Ev@plt>:
  400b30:	ff 25 1a 15 20 00    	jmpq   *0x20151a(%rip)        # 602050 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
  400b36:	68 07 00 00 00       	pushq  $0x7
  400b3b:	e9 70 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b40 <_ZNSo9_M_insertIdEERSoT_@plt>:
  400b40:	ff 25 12 15 20 00    	jmpq   *0x201512(%rip)        # 602058 <_ZNSo9_M_insertIdEERSoT_@GLIBCXX_3.4.9>
  400b46:	68 08 00 00 00       	pushq  $0x8
  400b4b:	e9 60 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b50 <calloc@plt>:
  400b50:	ff 25 0a 15 20 00    	jmpq   *0x20150a(%rip)        # 602060 <calloc@GLIBC_2.2.5>
  400b56:	68 09 00 00 00       	pushq  $0x9
  400b5b:	e9 50 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b60 <__cxa_atexit@plt>:
  400b60:	ff 25 02 15 20 00    	jmpq   *0x201502(%rip)        # 602068 <__cxa_atexit@GLIBC_2.2.5>
  400b66:	68 0a 00 00 00       	pushq  $0xa
  400b6b:	e9 40 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b70 <malloc@plt>:
  400b70:	ff 25 fa 14 20 00    	jmpq   *0x2014fa(%rip)        # 602070 <malloc@GLIBC_2.2.5>
  400b76:	68 0b 00 00 00       	pushq  $0xb
  400b7b:	e9 30 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b80 <__libc_start_main@plt>:
  400b80:	ff 25 f2 14 20 00    	jmpq   *0x2014f2(%rip)        # 602078 <__libc_start_main@GLIBC_2.2.5>
  400b86:	68 0c 00 00 00       	pushq  $0xc
  400b8b:	e9 20 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
  400b90:	ff 25 ea 14 20 00    	jmpq   *0x2014ea(%rip)        # 602080 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
  400b96:	68 0d 00 00 00       	pushq  $0xd
  400b9b:	e9 10 ff ff ff       	jmpq   400ab0 <.plt>

0000000000400ba0 <_ZNSt8ios_base4InitC1Ev@plt>:
  400ba0:	ff 25 e2 14 20 00    	jmpq   *0x2014e2(%rip)        # 602088 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
  400ba6:	68 0e 00 00 00       	pushq  $0xe
  400bab:	e9 00 ff ff ff       	jmpq   400ab0 <.plt>

Disassembly of section .text:

0000000000400bb0 <main>:
  400bb0:	55                   	push   %rbp
  400bb1:	bf 00 01 00 00       	mov    $0x100,%edi
  400bb6:	48 89 e5             	mov    %rsp,%rbp
  400bb9:	41 57                	push   %r15
  400bbb:	45 31 ff             	xor    %r15d,%r15d
  400bbe:	41 56                	push   %r14
  400bc0:	41 55                	push   %r13
  400bc2:	41 54                	push   %r12
  400bc4:	53                   	push   %rbx
  400bc5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400bc9:	48 81 ec e0 00 00 00 	sub    $0xe0,%rsp
  400bd0:	c5 f8 28 05 18 09 00 	vmovaps 0x918(%rip),%xmm0        # 4014f0 <__dso_handle+0xc8>
  400bd7:	00 
  400bd8:	c5 f8 29 84 24 80 00 	vmovaps %xmm0,0x80(%rsp)
  400bdf:	00 00 
  400be1:	c5 f8 28 05 17 09 00 	vmovaps 0x917(%rip),%xmm0        # 401500 <__dso_handle+0xd8>
  400be8:	00 
  400be9:	c5 f8 29 84 24 a0 00 	vmovaps %xmm0,0xa0(%rsp)
  400bf0:	00 00 
  400bf2:	c5 f8 28 05 16 09 00 	vmovaps 0x916(%rip),%xmm0        # 401510 <__dso_handle+0xe8>
  400bf9:	00 
  400bfa:	c5 f8 29 84 24 c0 00 	vmovaps %xmm0,0xc0(%rsp)
  400c01:	00 00 
  400c03:	e8 68 ff ff ff       	callq  400b70 <malloc@plt>
  400c08:	be 04 00 00 00       	mov    $0x4,%esi
  400c0d:	bf 10 00 00 00       	mov    $0x10,%edi
  400c12:	49 89 c4             	mov    %rax,%r12
  400c15:	e8 36 ff ff ff       	callq  400b50 <calloc@plt>
  400c1a:	be 04 00 00 00       	mov    $0x4,%esi
  400c1f:	bf 10 00 00 00       	mov    $0x10,%edi
  400c24:	49 89 c6             	mov    %rax,%r14
  400c27:	e8 24 ff ff ff       	callq  400b50 <calloc@plt>
  400c2c:	be 04 00 00 00       	mov    $0x4,%esi
  400c31:	bf 10 00 00 00       	mov    $0x10,%edi
  400c36:	48 89 c3             	mov    %rax,%rbx
  400c39:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c3e:	e8 0d ff ff ff       	callq  400b50 <calloc@plt>
  400c43:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400c47:	48 8d 94 24 c0 00 00 	lea    0xc0(%rsp),%rdx
  400c4e:	00 
  400c4f:	c5 f8 28 25 39 08 00 	vmovaps 0x839(%rip),%xmm4        # 401490 <__dso_handle+0x68>
  400c56:	00 
  400c57:	48 8d b4 24 a0 00 00 	lea    0xa0(%rsp),%rsi
  400c5e:	00 
  400c5f:	48 8d bc 24 80 00 00 	lea    0x80(%rsp),%rdi
  400c66:	00 
  400c67:	49 89 c1             	mov    %rax,%r9
  400c6a:	49 89 d8             	mov    %rbx,%r8
  400c6d:	c5 f8 28 2d 2b 08 00 	vmovaps 0x82b(%rip),%xmm5        # 4014a0 <__dso_handle+0x78>
  400c74:	00 
  400c75:	4c 89 f1             	mov    %r14,%rcx
  400c78:	c7 44 24 08 04 00 00 	movl   $0x4,0x8(%rsp)
  400c7f:	00 
  400c80:	4c 89 24 24          	mov    %r12,(%rsp)
  400c84:	c5 f8 28 35 24 08 00 	vmovaps 0x824(%rip),%xmm6        # 4014b0 <__dso_handle+0x88>
  400c8b:	00 
  400c8c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c91:	c5 f8 28 0d 27 08 00 	vmovaps 0x827(%rip),%xmm1        # 4014c0 <__dso_handle+0x98>
  400c98:	00 
  400c99:	c5 f8 28 15 2f 08 00 	vmovaps 0x82f(%rip),%xmm2        # 4014d0 <__dso_handle+0xa8>
  400ca0:	00 
  400ca1:	c5 f8 28 1d 37 08 00 	vmovaps 0x837(%rip),%xmm3        # 4014e0 <__dso_handle+0xb8>
  400ca8:	00 
  400ca9:	c4 c1 78 11 24 24    	vmovups %xmm4,(%r12)
  400caf:	c4 c1 78 11 6c 24 10 	vmovups %xmm5,0x10(%r12)
  400cb6:	c4 c1 78 11 74 24 20 	vmovups %xmm6,0x20(%r12)
  400cbd:	c4 c1 78 11 44 24 30 	vmovups %xmm0,0x30(%r12)
  400cc4:	c4 c1 78 11 4c 24 40 	vmovups %xmm1,0x40(%r12)
  400ccb:	c4 c1 78 11 54 24 50 	vmovups %xmm2,0x50(%r12)
  400cd2:	c4 c1 78 11 5c 24 60 	vmovups %xmm3,0x60(%r12)
  400cd9:	c4 c1 78 11 44 24 70 	vmovups %xmm0,0x70(%r12)
  400ce0:	c4 c1 78 11 84 24 80 	vmovups %xmm0,0x80(%r12)
  400ce7:	00 00 00 
  400cea:	c4 c1 78 11 b4 24 90 	vmovups %xmm6,0x90(%r12)
  400cf1:	00 00 00 
  400cf4:	c4 c1 78 11 ac 24 a0 	vmovups %xmm5,0xa0(%r12)
  400cfb:	00 00 00 
  400cfe:	c4 c1 78 11 a4 24 b0 	vmovups %xmm4,0xb0(%r12)
  400d05:	00 00 00 
  400d08:	c4 c1 78 11 84 24 c0 	vmovups %xmm0,0xc0(%r12)
  400d0f:	00 00 00 
  400d12:	c4 c1 78 11 9c 24 d0 	vmovups %xmm3,0xd0(%r12)
  400d19:	00 00 00 
  400d1c:	c4 c1 78 11 94 24 e0 	vmovups %xmm2,0xe0(%r12)
  400d23:	00 00 00 
  400d26:	c4 c1 78 11 8c 24 f0 	vmovups %xmm1,0xf0(%r12)
  400d2d:	00 00 00 
  400d30:	e8 4b 03 00 00       	callq  401080 <_Z6kernelPfS_S_S_S_S_S_i>
  400d35:	be 30 14 40 00       	mov    $0x401430,%esi
  400d3a:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400d3f:	e8 dc fd ff ff       	callq  400b20 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  400d44:	4b 8d 1c 3e          	lea    (%r14,%r15,1),%rbx
  400d48:	45 31 ed             	xor    %r13d,%r13d
  400d4b:	c4 a1 7a 10 04 2b    	vmovss (%rbx,%r13,1),%xmm0
  400d51:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400d56:	49 83 c5 04          	add    $0x4,%r13
  400d5a:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400d5e:	e8 dd fd ff ff       	callq  400b40 <_ZNSo9_M_insertIdEERSoT_@plt>
  400d63:	48 8d 74 24 7d       	lea    0x7d(%rsp),%rsi
  400d68:	ba 01 00 00 00       	mov    $0x1,%edx
  400d6d:	48 89 c7             	mov    %rax,%rdi
  400d70:	c6 44 24 7d 09       	movb   $0x9,0x7d(%rsp)
  400d75:	e8 16 fe ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400d7a:	49 83 fd 10          	cmp    $0x10,%r13
  400d7e:	75 cb                	jne    400d4b <main+0x19b>
  400d80:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
  400d85:	ba 01 00 00 00       	mov    $0x1,%edx
  400d8a:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400d8f:	49 83 c7 10          	add    $0x10,%r15
  400d93:	c6 44 24 20 0a       	movb   $0xa,0x20(%rsp)
  400d98:	e8 f3 fd ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400d9d:	49 83 ff 40          	cmp    $0x40,%r15
  400da1:	75 a1                	jne    400d44 <main+0x194>
  400da3:	be 4c 14 40 00       	mov    $0x40144c,%esi
  400da8:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400dad:	45 31 ff             	xor    %r15d,%r15d
  400db0:	e8 6b fd ff ff       	callq  400b20 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  400db5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400dba:	45 31 ed             	xor    %r13d,%r13d
  400dbd:	4a 8d 1c 38          	lea    (%rax,%r15,1),%rbx
  400dc1:	c4 a1 7a 10 04 2b    	vmovss (%rbx,%r13,1),%xmm0
  400dc7:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400dcc:	49 83 c5 04          	add    $0x4,%r13
  400dd0:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400dd4:	e8 67 fd ff ff       	callq  400b40 <_ZNSo9_M_insertIdEERSoT_@plt>
  400dd9:	48 8d 74 24 7e       	lea    0x7e(%rsp),%rsi
  400dde:	ba 01 00 00 00       	mov    $0x1,%edx
  400de3:	48 89 c7             	mov    %rax,%rdi
  400de6:	c6 44 24 7e 09       	movb   $0x9,0x7e(%rsp)
  400deb:	e8 a0 fd ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400df0:	49 83 fd 10          	cmp    $0x10,%r13
  400df4:	75 cb                	jne    400dc1 <main+0x211>
  400df6:	48 8d 74 24 40       	lea    0x40(%rsp),%rsi
  400dfb:	ba 01 00 00 00       	mov    $0x1,%edx
  400e00:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400e05:	49 83 c7 10          	add    $0x10,%r15
  400e09:	c6 44 24 40 0a       	movb   $0xa,0x40(%rsp)
  400e0e:	e8 7d fd ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400e13:	49 83 ff 40          	cmp    $0x40,%r15
  400e17:	75 9c                	jne    400db5 <main+0x205>
  400e19:	be 68 14 40 00       	mov    $0x401468,%esi
  400e1e:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400e23:	45 31 ff             	xor    %r15d,%r15d
  400e26:	e8 f5 fc ff ff       	callq  400b20 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
  400e2b:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400e30:	45 31 ed             	xor    %r13d,%r13d
  400e33:	4a 8d 1c 38          	lea    (%rax,%r15,1),%rbx
  400e37:	c4 a1 7a 10 04 2b    	vmovss (%rbx,%r13,1),%xmm0
  400e3d:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400e42:	49 83 c5 04          	add    $0x4,%r13
  400e46:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  400e4a:	e8 f1 fc ff ff       	callq  400b40 <_ZNSo9_M_insertIdEERSoT_@plt>
  400e4f:	48 8d 74 24 7f       	lea    0x7f(%rsp),%rsi
  400e54:	ba 01 00 00 00       	mov    $0x1,%edx
  400e59:	48 89 c7             	mov    %rax,%rdi
  400e5c:	c6 44 24 7f 09       	movb   $0x9,0x7f(%rsp)
  400e61:	e8 2a fd ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400e66:	49 83 fd 10          	cmp    $0x10,%r13
  400e6a:	75 cb                	jne    400e37 <main+0x287>
  400e6c:	48 8d 74 24 60       	lea    0x60(%rsp),%rsi
  400e71:	ba 01 00 00 00       	mov    $0x1,%edx
  400e76:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400e7b:	49 83 c7 10          	add    $0x10,%r15
  400e7f:	c6 44 24 60 0a       	movb   $0xa,0x60(%rsp)
  400e84:	e8 07 fd ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  400e89:	49 83 ff 40          	cmp    $0x40,%r15
  400e8d:	75 9c                	jne    400e2b <main+0x27b>
  400e8f:	4c 89 e7             	mov    %r12,%rdi
  400e92:	e8 29 fc ff ff       	callq  400ac0 <free@plt>
  400e97:	4c 89 f7             	mov    %r14,%rdi
  400e9a:	e8 21 fc ff ff       	callq  400ac0 <free@plt>
  400e9f:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  400ea4:	e8 17 fc ff ff       	callq  400ac0 <free@plt>
  400ea9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400eae:	e8 0d fc ff ff       	callq  400ac0 <free@plt>
  400eb3:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  400eb7:	31 c0                	xor    %eax,%eax
  400eb9:	5b                   	pop    %rbx
  400eba:	41 5c                	pop    %r12
  400ebc:	41 5d                	pop    %r13
  400ebe:	41 5e                	pop    %r14
  400ec0:	41 5f                	pop    %r15
  400ec2:	5d                   	pop    %rbp
  400ec3:	c3                   	retq   
  400ec4:	66 90                	xchg   %ax,%ax
  400ec6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ecd:	00 00 00 

0000000000400ed0 <_GLOBAL__sub_I__Z11kernelSaucePf>:
  400ed0:	48 83 ec 08          	sub    $0x8,%rsp
  400ed4:	bf b1 21 60 00       	mov    $0x6021b1,%edi
  400ed9:	e8 c2 fc ff ff       	callq  400ba0 <_ZNSt8ios_base4InitC1Ev@plt>
  400ede:	ba 28 14 40 00       	mov    $0x401428,%edx
  400ee3:	be b1 21 60 00       	mov    $0x6021b1,%esi
  400ee8:	bf 30 0b 40 00       	mov    $0x400b30,%edi
  400eed:	48 83 c4 08          	add    $0x8,%rsp
  400ef1:	e9 6a fc ff ff       	jmpq   400b60 <__cxa_atexit@plt>

0000000000400ef6 <_start>:
  400ef6:	31 ed                	xor    %ebp,%ebp
  400ef8:	49 89 d1             	mov    %rdx,%r9
  400efb:	5e                   	pop    %rsi
  400efc:	48 89 e2             	mov    %rsp,%rdx
  400eff:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  400f03:	50                   	push   %rax
  400f04:	54                   	push   %rsp
  400f05:	49 c7 c0 10 14 40 00 	mov    $0x401410,%r8
  400f0c:	48 c7 c1 a0 13 40 00 	mov    $0x4013a0,%rcx
  400f13:	48 c7 c7 b0 0b 40 00 	mov    $0x400bb0,%rdi
  400f1a:	e8 61 fc ff ff       	callq  400b80 <__libc_start_main@plt>
  400f1f:	f4                   	hlt    

0000000000400f20 <deregister_tm_clones>:
  400f20:	b8 9f 20 60 00       	mov    $0x60209f,%eax
  400f25:	55                   	push   %rbp
  400f26:	48 2d 98 20 60 00    	sub    $0x602098,%rax
  400f2c:	48 83 f8 0e          	cmp    $0xe,%rax
  400f30:	48 89 e5             	mov    %rsp,%rbp
  400f33:	77 02                	ja     400f37 <deregister_tm_clones+0x17>
  400f35:	5d                   	pop    %rbp
  400f36:	c3                   	retq   
  400f37:	b8 00 00 00 00       	mov    $0x0,%eax
  400f3c:	48 85 c0             	test   %rax,%rax
  400f3f:	74 f4                	je     400f35 <deregister_tm_clones+0x15>
  400f41:	5d                   	pop    %rbp
  400f42:	bf 98 20 60 00       	mov    $0x602098,%edi
  400f47:	ff e0                	jmpq   *%rax
  400f49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400f50 <register_tm_clones>:
  400f50:	b8 98 20 60 00       	mov    $0x602098,%eax
  400f55:	55                   	push   %rbp
  400f56:	48 2d 98 20 60 00    	sub    $0x602098,%rax
  400f5c:	48 c1 f8 03          	sar    $0x3,%rax
  400f60:	48 89 e5             	mov    %rsp,%rbp
  400f63:	48 89 c2             	mov    %rax,%rdx
  400f66:	48 c1 ea 3f          	shr    $0x3f,%rdx
  400f6a:	48 01 d0             	add    %rdx,%rax
  400f6d:	48 d1 f8             	sar    %rax
  400f70:	75 02                	jne    400f74 <register_tm_clones+0x24>
  400f72:	5d                   	pop    %rbp
  400f73:	c3                   	retq   
  400f74:	ba 00 00 00 00       	mov    $0x0,%edx
  400f79:	48 85 d2             	test   %rdx,%rdx
  400f7c:	74 f4                	je     400f72 <register_tm_clones+0x22>
  400f7e:	5d                   	pop    %rbp
  400f7f:	48 89 c6             	mov    %rax,%rsi
  400f82:	bf 98 20 60 00       	mov    $0x602098,%edi
  400f87:	ff e2                	jmpq   *%rdx
  400f89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400f90 <__do_global_dtors_aux>:
  400f90:	80 3d 19 12 20 00 00 	cmpb   $0x0,0x201219(%rip)        # 6021b0 <completed.6355>
  400f97:	75 11                	jne    400faa <__do_global_dtors_aux+0x1a>
  400f99:	55                   	push   %rbp
  400f9a:	48 89 e5             	mov    %rsp,%rbp
  400f9d:	e8 7e ff ff ff       	callq  400f20 <deregister_tm_clones>
  400fa2:	5d                   	pop    %rbp
  400fa3:	c6 05 06 12 20 00 01 	movb   $0x1,0x201206(%rip)        # 6021b0 <completed.6355>
  400faa:	f3 c3                	repz retq 
  400fac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400fb0 <frame_dummy>:
  400fb0:	48 83 3d 38 0d 20 00 	cmpq   $0x0,0x200d38(%rip)        # 601cf0 <__JCR_END__>
  400fb7:	00 
  400fb8:	74 1e                	je     400fd8 <frame_dummy+0x28>
  400fba:	b8 00 00 00 00       	mov    $0x0,%eax
  400fbf:	48 85 c0             	test   %rax,%rax
  400fc2:	74 14                	je     400fd8 <frame_dummy+0x28>
  400fc4:	55                   	push   %rbp
  400fc5:	bf f0 1c 60 00       	mov    $0x601cf0,%edi
  400fca:	48 89 e5             	mov    %rsp,%rbp
  400fcd:	ff d0                	callq  *%rax
  400fcf:	5d                   	pop    %rbp
  400fd0:	e9 7b ff ff ff       	jmpq   400f50 <register_tm_clones>
  400fd5:	0f 1f 00             	nopl   (%rax)
  400fd8:	e9 73 ff ff ff       	jmpq   400f50 <register_tm_clones>
  400fdd:	0f 1f 00             	nopl   (%rax)

0000000000400fe0 <_Z11kernelSaucePf>:
  400fe0:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400fe4:	c5 f8 28 25 a4 04 00 	vmovaps 0x4a4(%rip),%xmm4        # 401490 <__dso_handle+0x68>
  400feb:	00 
  400fec:	c5 f8 28 2d ac 04 00 	vmovaps 0x4ac(%rip),%xmm5        # 4014a0 <__dso_handle+0x78>
  400ff3:	00 
  400ff4:	c5 f8 28 35 b4 04 00 	vmovaps 0x4b4(%rip),%xmm6        # 4014b0 <__dso_handle+0x88>
  400ffb:	00 
  400ffc:	c5 f8 28 0d bc 04 00 	vmovaps 0x4bc(%rip),%xmm1        # 4014c0 <__dso_handle+0x98>
  401003:	00 
  401004:	c5 f8 28 15 c4 04 00 	vmovaps 0x4c4(%rip),%xmm2        # 4014d0 <__dso_handle+0xa8>
  40100b:	00 
  40100c:	c5 f8 28 1d cc 04 00 	vmovaps 0x4cc(%rip),%xmm3        # 4014e0 <__dso_handle+0xb8>
  401013:	00 
  401014:	c5 f8 29 27          	vmovaps %xmm4,(%rdi)
  401018:	c5 f8 29 6f 10       	vmovaps %xmm5,0x10(%rdi)
  40101d:	c5 f8 29 77 20       	vmovaps %xmm6,0x20(%rdi)
  401022:	c5 f8 29 47 30       	vmovaps %xmm0,0x30(%rdi)
  401027:	c5 f8 29 4f 40       	vmovaps %xmm1,0x40(%rdi)
  40102c:	c5 f8 29 57 50       	vmovaps %xmm2,0x50(%rdi)
  401031:	c5 f8 29 5f 60       	vmovaps %xmm3,0x60(%rdi)
  401036:	c5 f8 29 47 70       	vmovaps %xmm0,0x70(%rdi)
  40103b:	c5 f8 29 87 80 00 00 	vmovaps %xmm0,0x80(%rdi)
  401042:	00 
  401043:	c5 f8 29 b7 90 00 00 	vmovaps %xmm6,0x90(%rdi)
  40104a:	00 
  40104b:	c5 f8 29 af a0 00 00 	vmovaps %xmm5,0xa0(%rdi)
  401052:	00 
  401053:	c5 f8 29 a7 b0 00 00 	vmovaps %xmm4,0xb0(%rdi)
  40105a:	00 
  40105b:	c5 f8 29 87 c0 00 00 	vmovaps %xmm0,0xc0(%rdi)
  401062:	00 
  401063:	c5 f8 29 9f d0 00 00 	vmovaps %xmm3,0xd0(%rdi)
  40106a:	00 
  40106b:	c5 f8 29 97 e0 00 00 	vmovaps %xmm2,0xe0(%rdi)
  401072:	00 
  401073:	c5 f8 29 8f f0 00 00 	vmovaps %xmm1,0xf0(%rdi)
  40107a:	00 
  40107b:	c3                   	retq   
  40107c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401080 <_Z6kernelPfS_S_S_S_S_S_i>:
  401080:	55                   	push   %rbp
  401081:	48 89 cd             	mov    %rcx,%rbp
  401084:	53                   	push   %rbx
  401085:	31 db                	xor    %ebx,%ebx
  401087:	48 83 ec 48          	sub    $0x48,%rsp
  40108b:	c5 f8 28 3f          	vmovaps (%rdi),%xmm7
  40108f:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  401094:	c4 e3 79 04 cf 55    	vpermilps $0x55,%xmm7,%xmm1
  40109a:	c5 f8 28 36          	vmovaps (%rsi),%xmm6
  40109e:	c4 e3 79 04 e7 00    	vpermilps $0x0,%xmm7,%xmm4
  4010a4:	c5 f8 28 2a          	vmovaps (%rdx),%xmm5
  4010a8:	c4 e3 79 04 de 00    	vpermilps $0x0,%xmm6,%xmm3
  4010ae:	c5 78 28 c1          	vmovaps %xmm1,%xmm8
  4010b2:	c4 e3 79 04 ce 55    	vpermilps $0x55,%xmm6,%xmm1
  4010b8:	c5 f8 28 40 50       	vmovaps 0x50(%rax),%xmm0
  4010bd:	c5 78 28 ec          	vmovaps %xmm4,%xmm13
  4010c1:	c5 78 28 38          	vmovaps (%rax),%xmm15
  4010c5:	c5 78 28 e4          	vmovaps %xmm4,%xmm12
  4010c9:	c5 78 28 c9          	vmovaps %xmm1,%xmm9
  4010cd:	c4 e3 79 04 cd 55    	vpermilps $0x55,%xmm5,%xmm1
  4010d3:	c5 78 28 70 10       	vmovaps 0x10(%rax),%xmm14
  4010d8:	c4 62 01 a8 29       	vfmadd213ps (%rcx),%xmm15,%xmm13
  4010dd:	c5 78 28 db          	vmovaps %xmm3,%xmm11
  4010e1:	c4 e3 79 04 d5 00    	vpermilps $0x0,%xmm5,%xmm2
  4010e7:	c4 62 09 a8 61 10    	vfmadd213ps 0x10(%rcx),%xmm14,%xmm12
  4010ed:	c5 78 28 d3          	vmovaps %xmm3,%xmm10
  4010f1:	c5 f8 29 0c 24       	vmovaps %xmm1,(%rsp)
  4010f6:	c4 42 01 a8 18       	vfmadd213ps (%r8),%xmm15,%xmm11
  4010fb:	c4 42 79 b8 e0       	vfmadd231ps %xmm8,%xmm0,%xmm12
  401100:	c4 42 09 a8 50 10    	vfmadd213ps 0x10(%r8),%xmm14,%xmm10
  401106:	c4 42 79 b8 d1       	vfmadd231ps %xmm9,%xmm0,%xmm10
  40110b:	c5 78 29 44 24 10    	vmovaps %xmm8,0x10(%rsp)
  401111:	c4 42 69 a8 39       	vfmadd213ps (%r9),%xmm2,%xmm15
  401116:	c4 42 69 a8 71 10    	vfmadd213ps 0x10(%r9),%xmm2,%xmm14
  40111c:	c5 78 29 4c 24 20    	vmovaps %xmm9,0x20(%rsp)
  401122:	c5 f8 28 48 40       	vmovaps 0x40(%rax),%xmm1
  401127:	c4 42 71 b8 e8       	vfmadd231ps %xmm8,%xmm1,%xmm13
  40112c:	c4 42 71 b8 d9       	vfmadd231ps %xmm9,%xmm1,%xmm11
  401131:	c5 78 28 c1          	vmovaps %xmm1,%xmm8
  401135:	c5 f8 28 0c 24       	vmovaps (%rsp),%xmm1
  40113a:	c4 62 01 98 c1       	vfmadd132ps %xmm1,%xmm15,%xmm8
  40113f:	c4 62 79 b8 f1       	vfmadd231ps %xmm1,%xmm0,%xmm14
  401144:	c4 63 79 04 fe aa    	vpermilps $0xaa,%xmm6,%xmm15
  40114a:	c4 e3 79 04 cf aa    	vpermilps $0xaa,%xmm7,%xmm1
  401150:	c5 78 28 88 80 00 00 	vmovaps 0x80(%rax),%xmm9
  401157:	00 
  401158:	c4 e3 79 04 c5 aa    	vpermilps $0xaa,%xmm5,%xmm0
  40115e:	c4 e3 79 04 ff ff    	vpermilps $0xff,%xmm7,%xmm7
  401164:	c4 42 01 b8 d9       	vfmadd231ps %xmm9,%xmm15,%xmm11
  401169:	c4 62 01 b8 90 90 00 	vfmadd231ps 0x90(%rax),%xmm15,%xmm10
  401170:	00 00 
  401172:	c4 e3 79 04 f6 ff    	vpermilps $0xff,%xmm6,%xmm6
  401178:	c4 42 71 b8 e9       	vfmadd231ps %xmm9,%xmm1,%xmm13
  40117d:	c4 62 71 b8 a0 90 00 	vfmadd231ps 0x90(%rax),%xmm1,%xmm12
  401184:	00 00 
  401186:	c4 e3 79 04 ed ff    	vpermilps $0xff,%xmm5,%xmm5
  40118c:	c4 62 79 b8 b0 90 00 	vfmadd231ps 0x90(%rax),%xmm0,%xmm14
  401193:	00 00 
  401195:	c4 62 39 98 c8       	vfmadd132ps %xmm0,%xmm8,%xmm9
  40119a:	c4 41 78 28 c6       	vmovaps %xmm14,%xmm8
  40119f:	c5 78 28 b0 c0 00 00 	vmovaps 0xc0(%rax),%xmm14
  4011a6:	00 
  4011a7:	c4 62 41 b8 a0 d0 00 	vfmadd231ps 0xd0(%rax),%xmm7,%xmm12
  4011ae:	00 00 
  4011b0:	c4 62 49 b8 90 d0 00 	vfmadd231ps 0xd0(%rax),%xmm6,%xmm10
  4011b7:	00 00 
  4011b9:	c4 62 51 b8 80 d0 00 	vfmadd231ps 0xd0(%rax),%xmm5,%xmm8
  4011c0:	00 00 
  4011c2:	c4 42 41 b8 ee       	vfmadd231ps %xmm14,%xmm7,%xmm13
  4011c7:	c5 78 29 29          	vmovaps %xmm13,(%rcx)
  4011cb:	48 63 4c 24 68       	movslq 0x68(%rsp),%rcx
  4011d0:	c4 42 49 b8 de       	vfmadd231ps %xmm14,%xmm6,%xmm11
  4011d5:	c4 42 51 b8 ce       	vfmadd231ps %xmm14,%xmm5,%xmm9
  4011da:	48 8d 14 8d 00 00 00 	lea    0x0(,%rcx,4),%rdx
  4011e1:	00 
  4011e2:	4c 8d 54 15 00       	lea    0x0(%rbp,%rdx,1),%r10
  4011e7:	49 8d 3c 10          	lea    (%r8,%rdx,1),%rdi
  4011eb:	49 8d 34 11          	lea    (%r9,%rdx,1),%rsi
  4011ef:	c4 41 78 29 22       	vmovaps %xmm12,(%r10)
  4011f4:	c4 41 78 29 18       	vmovaps %xmm11,(%r8)
  4011f9:	c5 78 28 dc          	vmovaps %xmm4,%xmm11
  4011fd:	c5 78 29 17          	vmovaps %xmm10,(%rdi)
  401201:	c5 78 28 d3          	vmovaps %xmm3,%xmm10
  401205:	c4 41 78 29 09       	vmovaps %xmm9,(%r9)
  40120a:	c5 78 29 06          	vmovaps %xmm8,(%rsi)
  40120e:	c5 78 28 40 30       	vmovaps 0x30(%rax),%xmm8
  401213:	c5 78 28 48 20       	vmovaps 0x20(%rax),%xmm9
  401218:	c4 e2 39 a8 65 30    	vfmadd213ps 0x30(%rbp),%xmm8,%xmm4
  40121e:	c4 c2 39 a8 58 30    	vfmadd213ps 0x30(%r8),%xmm8,%xmm3
  401224:	c4 62 31 a8 5d 20    	vfmadd213ps 0x20(%rbp),%xmm9,%xmm11
  40122a:	c4 42 31 a8 50 20    	vfmadd213ps 0x20(%r8),%xmm9,%xmm10
  401230:	c5 78 28 60 60       	vmovaps 0x60(%rax),%xmm12
  401235:	c4 42 69 a8 49 20    	vfmadd213ps 0x20(%r9),%xmm2,%xmm9
  40123b:	c4 c2 39 a8 51 30    	vfmadd213ps 0x30(%r9),%xmm8,%xmm2
  401241:	c5 78 28 6c 24 10    	vmovaps 0x10(%rsp),%xmm13
  401247:	c5 78 28 40 70       	vmovaps 0x70(%rax),%xmm8
  40124c:	c4 42 19 b8 dd       	vfmadd231ps %xmm13,%xmm12,%xmm11
  401251:	c4 c2 39 b8 e5       	vfmadd231ps %xmm13,%xmm8,%xmm4
  401256:	c5 78 28 6c 24 20    	vmovaps 0x20(%rsp),%xmm13
  40125c:	c4 42 19 b8 d5       	vfmadd231ps %xmm13,%xmm12,%xmm10
  401261:	c4 c2 39 b8 dd       	vfmadd231ps %xmm13,%xmm8,%xmm3
  401266:	c5 78 28 2c 24       	vmovaps (%rsp),%xmm13
  40126b:	c5 78 28 b0 a0 00 00 	vmovaps 0xa0(%rax),%xmm14
  401272:	00 
  401273:	c4 42 19 b8 cd       	vfmadd231ps %xmm13,%xmm12,%xmm9
  401278:	c4 c2 39 b8 d5       	vfmadd231ps %xmm13,%xmm8,%xmm2
  40127d:	c5 78 28 a8 b0 00 00 	vmovaps 0xb0(%rax),%xmm13
  401284:	00 
  401285:	c4 42 71 b8 de       	vfmadd231ps %xmm14,%xmm1,%xmm11
  40128a:	c4 42 01 b8 d6       	vfmadd231ps %xmm14,%xmm15,%xmm10
  40128f:	c4 42 79 b8 ce       	vfmadd231ps %xmm14,%xmm0,%xmm9
  401294:	c4 c2 71 b8 e5       	vfmadd231ps %xmm13,%xmm1,%xmm4
  401299:	c5 78 28 b0 e0 00 00 	vmovaps 0xe0(%rax),%xmm14
  4012a0:	00 
  4012a1:	c4 c2 01 b8 dd       	vfmadd231ps %xmm13,%xmm15,%xmm3
  4012a6:	c4 c2 79 b8 d5       	vfmadd231ps %xmm13,%xmm0,%xmm2
  4012ab:	c5 78 28 a8 f0 00 00 	vmovaps 0xf0(%rax),%xmm13
  4012b2:	00 
  4012b3:	49 8d 04 12          	lea    (%r10,%rdx,1),%rax
  4012b7:	c4 42 41 b8 de       	vfmadd231ps %xmm14,%xmm7,%xmm11
  4012bc:	c4 42 49 b8 d6       	vfmadd231ps %xmm14,%xmm6,%xmm10
  4012c1:	c4 62 31 98 f5       	vfmadd132ps %xmm5,%xmm9,%xmm14
  4012c6:	c5 78 29 18          	vmovaps %xmm11,(%rax)
  4012ca:	c4 c2 59 98 fd       	vfmadd132ps %xmm13,%xmm4,%xmm7
  4012cf:	c4 c2 61 98 f5       	vfmadd132ps %xmm13,%xmm3,%xmm6
  4012d4:	c4 c2 69 98 ed       	vfmadd132ps %xmm13,%xmm2,%xmm5
  4012d9:	c5 f8 29 3c 88       	vmovaps %xmm7,(%rax,%rcx,4)
  4012de:	48 8d 04 17          	lea    (%rdi,%rdx,1),%rax
  4012e2:	48 01 f2             	add    %rsi,%rdx
  4012e5:	c5 78 29 10          	vmovaps %xmm10,(%rax)
  4012e9:	c5 f8 29 34 88       	vmovaps %xmm6,(%rax,%rcx,4)
  4012ee:	c5 78 29 32          	vmovaps %xmm14,(%rdx)
  4012f2:	c5 f8 29 2c 8a       	vmovaps %xmm5,(%rdx,%rcx,4)
  4012f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4012fe:	00 00 
  401300:	c5 fa 10 44 1d 00    	vmovss 0x0(%rbp,%rbx,1),%xmm0
  401306:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  40130b:	48 83 c3 04          	add    $0x4,%rbx
  40130f:	c5 f8 5a c0          	vcvtps2pd %xmm0,%xmm0
  401313:	e8 28 f8 ff ff       	callq  400b40 <_ZNSo9_M_insertIdEERSoT_@plt>
  401318:	48 8d 74 24 3f       	lea    0x3f(%rsp),%rsi
  40131d:	ba 01 00 00 00       	mov    $0x1,%edx
  401322:	48 89 c7             	mov    %rax,%rdi
  401325:	c6 44 24 3f 09       	movb   $0x9,0x3f(%rsp)
  40132a:	e8 61 f8 ff ff       	callq  400b90 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
  40132f:	48 83 fb 40          	cmp    $0x40,%rbx
  401333:	75 cb                	jne    401300 <_Z6kernelPfS_S_S_S_S_S_i+0x280>
  401335:	48 8b 05 64 0d 20 00 	mov    0x200d64(%rip),%rax        # 6020a0 <_ZSt4cout@@GLIBCXX_3.4>
  40133c:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  401340:	48 8b 98 90 21 60 00 	mov    0x602190(%rax),%rbx
  401347:	48 85 db             	test   %rbx,%rbx
  40134a:	74 44                	je     401390 <_Z6kernelPfS_S_S_S_S_S_i+0x310>
  40134c:	80 7b 38 00          	cmpb   $0x0,0x38(%rbx)
  401350:	74 26                	je     401378 <_Z6kernelPfS_S_S_S_S_S_i+0x2f8>
  401352:	0f b6 43 43          	movzbl 0x43(%rbx),%eax
  401356:	0f be f0             	movsbl %al,%esi
  401359:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  40135e:	e8 ad f7 ff ff       	callq  400b10 <_ZNSo3putEc@plt>
  401363:	48 89 c7             	mov    %rax,%rdi
  401366:	e8 95 f7 ff ff       	callq  400b00 <_ZNSo5flushEv@plt>
  40136b:	48 83 c4 48          	add    $0x48,%rsp
  40136f:	5b                   	pop    %rbx
  401370:	5d                   	pop    %rbp
  401371:	c3                   	retq   
  401372:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401378:	48 89 df             	mov    %rbx,%rdi
  40137b:	e8 70 f7 ff ff       	callq  400af0 <_ZNKSt5ctypeIcE13_M_widen_initEv@plt>
  401380:	48 8b 03             	mov    (%rbx),%rax
  401383:	be 0a 00 00 00       	mov    $0xa,%esi
  401388:	48 89 df             	mov    %rbx,%rdi
  40138b:	ff 50 30             	callq  *0x30(%rax)
  40138e:	eb c6                	jmp    401356 <_Z6kernelPfS_S_S_S_S_S_i+0x2d6>
  401390:	e8 3b f7 ff ff       	callq  400ad0 <_ZSt16__throw_bad_castv@plt>
  401395:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40139c:	00 00 00 
  40139f:	90                   	nop

00000000004013a0 <__libc_csu_init>:
  4013a0:	41 57                	push   %r15
  4013a2:	41 89 ff             	mov    %edi,%r15d
  4013a5:	41 56                	push   %r14
  4013a7:	49 89 f6             	mov    %rsi,%r14
  4013aa:	41 55                	push   %r13
  4013ac:	49 89 d5             	mov    %rdx,%r13
  4013af:	41 54                	push   %r12
  4013b1:	4c 8d 25 20 09 20 00 	lea    0x200920(%rip),%r12        # 601cd8 <__frame_dummy_init_array_entry>
  4013b8:	55                   	push   %rbp
  4013b9:	48 8d 2d 28 09 20 00 	lea    0x200928(%rip),%rbp        # 601ce8 <__init_array_end>
  4013c0:	53                   	push   %rbx
  4013c1:	4c 29 e5             	sub    %r12,%rbp
  4013c4:	31 db                	xor    %ebx,%ebx
  4013c6:	48 c1 fd 03          	sar    $0x3,%rbp
  4013ca:	48 83 ec 08          	sub    $0x8,%rsp
  4013ce:	e8 bd f6 ff ff       	callq  400a90 <_init>
  4013d3:	48 85 ed             	test   %rbp,%rbp
  4013d6:	74 1e                	je     4013f6 <__libc_csu_init+0x56>
  4013d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4013df:	00 
  4013e0:	4c 89 ea             	mov    %r13,%rdx
  4013e3:	4c 89 f6             	mov    %r14,%rsi
  4013e6:	44 89 ff             	mov    %r15d,%edi
  4013e9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4013ed:	48 83 c3 01          	add    $0x1,%rbx
  4013f1:	48 39 eb             	cmp    %rbp,%rbx
  4013f4:	75 ea                	jne    4013e0 <__libc_csu_init+0x40>
  4013f6:	48 83 c4 08          	add    $0x8,%rsp
  4013fa:	5b                   	pop    %rbx
  4013fb:	5d                   	pop    %rbp
  4013fc:	41 5c                	pop    %r12
  4013fe:	41 5d                	pop    %r13
  401400:	41 5e                	pop    %r14
  401402:	41 5f                	pop    %r15
  401404:	c3                   	retq   
  401405:	90                   	nop
  401406:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40140d:	00 00 00 

0000000000401410 <__libc_csu_fini>:
  401410:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401414 <_fini>:
  401414:	48 83 ec 08          	sub    $0x8,%rsp
  401418:	48 83 c4 08          	add    $0x8,%rsp
  40141c:	c3                   	retq   
