# DWP CMG RAG Comparative Evaluation Report

This report provides the non-truncated outputs of the three evaluation configurations on the 3 core policy questions.

## Comparison Matrix

### [QUESTION 1] Explain how the 25% income variance rule applies to a paying parent's gross weekly income.

#### Retrieved Contexts:
- **Paragraph 19012** (Score: 0.4713, Source: `cmdmg-vol2-maintenance-calculations-chapters-17-to-26.pdf`):
  > This regulation is intended for NRPs who have regular earnings (the level of earnings is broadly consistent in each pay period) and who are receiving this amount on a regular basis (for example, the client is paid weekly/fortnightly etc. as opposed to on a fluctuating basis). The DM must be satisfied that this arrangement is likely to continue for the foreseeable future. Note: there is no minimum period of time which should be used to define the "foreseeable future". Instead DMs should consider the information available to determine whether the NRP's employment is reflective of the legislative intent. Example 1: Income is consistent - regular settled pattern Emma makes an application for child maintenance against Paul in respect of qualifying child Lucy. The initial maintenance calculation is effective from 21 September 2018 using an historic income amount of 21,000 (403.85 per week) for the latest available tax year 2016 - 2017. On 13 May 2019 Paul calls the CMS to advise of a change to his circumstances. He explains that he lost his job in April but started working for a new employer from ^ May 2019. Paul explains that his new job is a permanent job on a long term contract and that he will receive a regular monthly salary. Paul provides a signed declaration that his new gross monthly income is 750. As the earnings are more than 25% lower than the historic income information already held and as they are expected to continue for the foreseeable future, Regulation 38 2(a) should be applied and the gross weekly income should be based on the weekly equivalent of the NRP's monthly salary. Note: the effective date of the change will be the date the change was notified to the CMS, irrespective of when the NRP started the new employment or when they started to receive income at the new rate. Example 2: Income is broadly consistent - regular settled pattern PWC Saira contacts the CMS and requests a change to current income (from historic income) on the 23 November 2018 and provides reasonable grounds to support the request. RTI is checked and shows a 25% increase in income has occurred. There has been no verbal contact with the NRP to understand the nature of their employment, no bonus payments are reported. The following RTI is held covering the increased period: 23 November 519 30 November 505 7 December 519 15 December 510 22 December 519 As the amount the NRP receives each week is broadly consistent it is considered as regular according to a settled pattern. The NRP's gross weekly income should be determined as the weekly equivalent of the amounts paid. No further verification with the employer or non-resident parent is required unless commission / bonus payments are reported that are paid with a different frequency from regular earnings. Example 3: Income fluctuates but has a regular settled pattern PWC Jack requests a change to current income (from historic income) on the 24 October 2018 and provides reasonable grounds to support the request. RTI is checked and shows a 25% increase in the NRP's income has occurred. There has been no verbal contact with the NRP to understand the nature of their employment and no bonus payments are reported. The following RTI is held covering the increased period: 23 October 430 30 October 500 6 November 435 14 November 500 21 November 429 28 November 505 The amounts the NRP receives fluctuate but those fluctuations and levels follow a regular settled pattern that are broadly consistent. The NRP's gross weekly income should be determined as the weekly equivalent of the amounts paid.

- **Paragraph 18001** (Score: 0.4599, Source: `cmdmg-vol2-maintenance-calculations-chapters-17-to-26.pdf`):
  > For the purposes of calculating child maintenance liability, an NRP's gross weekly income is determined on the basis of historic income or current income. The preference is to use historic income, which is derived from figures held by HMRC. The historic income data must be used unless 1. current income differs from historic income by at least 25%1 2. no historic income is available2, or 3. CMS has been unable to request or obtain the required information3. 1 CSMC Regs 2012, reg 34(2)(a); 2 reg 34(2)(b); 3 reg 34(2)(c) Note: For further information on Current Income see Chapters 19 & 20. This guidance explains What is historic income

- **Paragraph 30000** (Score: 0.4372, Source: `volume-3-variations-chapters-27-36.pdf`):
  > x 0.6= 18,000). As John's projected income (25,000) exceeds 60% of his gross income (18,000) the difference of 7,000 (25,000 - 18,000= 7,000) can be applied to the variation. This will increase John's income by 7,000 for the purposes of the child maintenance calculation.

#### Comparative Outputs:
##### Configuration A: Tuned Model (No RAG)
```text
The 25% Income Variance Rule is used for calculating an employee’s net wage rate that they must pay in tax and NIC as per s.76 of ETSA (for individuals) or s.39(1)(b) of the Tax Act (for businesses). It takes into account their total monthly wages less any payroll deductions including PAYE contributions which include:

1. Employment allowances paid out from HMRC
2. Any amount set aside on behalf of the taxpayer by HMRC under section 84T of the Companies Acts relating to employees who have been laid off
3. The amounts normally deducted at source from earnings for pensionable scheme members such as savings group schemes and investment portfolios. These figures may be reduced if necessary due to personal circumstances related to the individual having retired and lost part-time work status. For example where someone has only worked part time then there might not have been enough earnings available to cover all payments made for retirement. This could reduce these pensions benefits payable. However this should not affect what other items can still be taken up with the employer when making their taxable salary figure - it will just need to be adjusted downwards after payment of PPSC to bring them back within the normal range.

Income Varity means anything over £200 excluding holiday allowance but below £300. If your gross wage exceeds £300, you are required to pay 25% VAT plus National Insurance
```

##### Configuration B: Base Model + RAG
```text
The 25% income variance rule states that if a paying parent's gross weekly income falls below 60% of their total household income, then any additional income above 60% would need to be added back into the original gross weekly income before applying the variation formula. In other words, the variation formula takes into account both the actual income and its percentage contribution compared to the overall household income.

In the case where a paying parent's gross weekly income falls below 60%, they may apply the variation formula without adding back any extra income because the excess income over 60% is essentially being subtracted directly from the original gross weekly income. Therefore, the variation formula only considers the net income after accounting for the proportion of the family's income contributed towards childcare costs.

So, under the 25% income variance rule, the variation formula calculates the adjusted gross weekly income based solely on the parents' combined incomes minus the portion of the family's income allocated towards childcare expenses. Any remaining unused parental income beyond 60% of the combined household income remains unchanged due to the variation formula's focus on the relative proportions rather than absolute values.
```

##### Configuration C: Tuned Model + RAG
```text
The 25% income variance rule states that if the total household income falls below 60%, then any additional income received above 60% of the previous month's income cannot affect the final calculated value of the family's disposable income. In other words, if the household income is less than 60%, the extra income earned during the last few months would only add up to 25% of the previous month’s income. Therefore, the excess income over 60% could either remain unchanged or decrease by 25%. However, the main point here is that even though the extra income may exceed 60% of the previous month's income, its impact on the overall calculations remains limited because the remaining portion after applying the percentage adjustment still needs to fall within the 60% threshold before adding anything else to the figure. So, regardless of what happens beyond the first three months, the extra income added stays under 60% so it doesn't count towards the full 25% multiplier.
```

---

### [QUESTION 2] Under the Child Maintenance (Enforcement) Act 2023, what powers does the department have regarding administrative Liability Orders?

#### Retrieved Contexts:
- **Paragraph 81005** (Score: 0.5972, Source: `volume-6-collection-and-enforcement-chapters-49-95.pdf`):
  > When a NRP doesn't pay the child maintenance they owe, CMS can apply for an order from a court that will allow enforcement action to be taken1. This is called a liability order and in Scotland is granted by the Sheriff2 in a Sheriff Court. It is a decree for payment, which certifies that a debt is legally due. The order will 1. define the period over which the debt has accrued 2. confirm the amount of debt that has accrued within that period and that it is legally due to be paid, and 3. confirm any expenses due to CMS. 1 CS Act 1991, s33 (1); 2 CS Act 1991, s33 (2) See chapter 83: Liability Orders - Scotland Pre Liability order actions Interim attachment1

- **Paragraph 56002** (Score: 0.5092, Source: `volume-6-collection-and-enforcement-chapters-49-95.pdf`):
  > Deduction orders are an administrative method of collecting child maintenance arrears, which means they can be imposed without court authorisation. There is no requirement for a liability order to be in place before a deduction order is imposed.

- **Paragraph 77005** (Score: 0.4832, Source: `volume-6-collection-and-enforcement-chapters-49-95.pdf`):
  > In deciding if an OFS application is appropriate DMs must consider if 1. a final charging order(s) has been granted and registered with HM Land Registry1. See Chapter 76: Charging Orders. 2. every effort has been made to resolve any IMA or a DMD that is or has been in place, as the debt may change if the NRP provides the information to complete a full maintenance calculation 3. there are outstanding reviews or appeals that may affect the arrears figure 4. where the majority of debt is owed to the PWC they have confirmed they wish this collected 5. all information in relation the application has been obtained by the DM from the NRP or CMS's information gathering resources 6. enforcement agent action has been taken and if no goods are available, a Nulla Bona must be received. See Chapter 74: Enforcement agents. 7. a DEO or deduction order can be imposed, even if this has previously been considered. See Chapter 55: DEOs and DERs & Chapter 56: Deduction orders. 8. records are held confirming all possible civil enforcement actions have been taken and concluded, or 9. an enforcement action has been considered and found not to be appropriate and the decisions have been fully recorded. See Chapter 75: Third party deduction orders & Chapter 73: Liability orders England and Wales. 1 CS Act 1991, s 36 & CP Rules 1998, Part 73

#### Comparative Outputs:
##### Configuration A: Tuned Model (No RAG)
```text
The Department of Social Protection has the power to issue an Administrative Liability Order under section 168(1)(b) of the Children's Law Enforcement and Regulation Act 2015. This allows them to take control over child support payments for those whose earnings or income do not meet their needs based on the previous annual assessment. The order can then specify whether any amount paid will be deducted from future claims or only if it is above £400 per month. If this applies, they need to make sure that the payment amounts go towards meeting eligible expenses such as accommodation costs, education fees etc. It also requires the person with liability obligations in relation to these payments to give notice of intent before making payments and must ensure all notices given out comply with legal requirements relating to disabled children. They should keep records related to outstanding liabilities until such time as satisfactory arrangements can be made. This includes ensuring that there are no further issues arising which may prevent reasonable adjustments being agreed between you and your employer or third party provider/organisations who provide financial assistance for the claimant. There are certain conditions requiring proof of incapacity including failing mental health tests; but otherwise this provision would likely apply even where disability was proven at employment starting date so care taken when drafting an actuary report - especially considering the extent of the evidence required and how much work had been done previously. To deal effectively with complaints against the agency concerned about compliance with the provisions set out below: Ensure that all relevant documents are kept
```

##### Configuration B: Base Model + RAG
```text
The Department of Work and Pensions (the "Department") has the power under the Child Maintenance (Enforcement) Act 2023 to impose Administrative Liability Orders on parents who fail to meet their obligations towards children. These orders require parents to make payments into the system at regular intervals until such time as these payments become payable. The powers include the ability to issue notices requiring parents to submit evidence about how much money should be paid each month, including whether the parent owes more than £1,000 per year, and issuing penalty notices when necessary. Additionally, the Department also has authority to enforce penalties against non-compliant parents up to a maximum fine of £10,000.
```

##### Configuration C: Tuned Model + RAG
```text
The Department of Work and Pensions (the "Department") has power under the Child Maintenance (Enforcement) Act 2023 to impose Administrative Liability Orders on parents who fail to comply with their obligations under the Children Act 1989 and the Child Maintenance Order Procedure Regulations 2023. These Orders require parents to make payments according to the terms set out in the Child Maintenance Order Procedure Regulations 2023. The Powers include imposing penalties up to £50,000 per day, suspending parental responsibility until further notice, and requiring parents to attend regular meetings at the Office of Social Protection (OSP). Additionally, the Power includes allowing the OSP to take possession of property owned by parents unless approved by the Secretary of State.
```

---

### [QUESTION 3] How are pension contributions treated when calculating a paying parent's child maintenance liability?

#### Retrieved Contexts:
- **Paragraph 18052** (Score: 0.6195, Source: `cmdmg-vol2-maintenance-calculations-chapters-17-to-26.pdf`):
  > Where the NRP receives an occupational pension from their deceased partner's former employers, as this income is taxable, it is included as income when calculating child maintenance. Chapter 19 - Current income - Employed Introduction

- **Paragraph 22010** (Score: 0.6171, Source: `cmdmg-vol2-maintenance-calculations-chapters-17-to-26.pdf`):
  > For the purposes of calculating the maintenance liability, the NRP's gross earnings figure is earnings minus pension contributions and non-taxable expenses. Table of non-taxable expenses definitions and examples

- **Paragraph 18026** (Score: 0.6035, Source: `cmdmg-vol2-maintenance-calculations-chapters-17-to-26.pdf`):
  > The income figure would not need to be adjusted to reflect shared care. In these circumstances, the maintenance liability is adjusted rather than the NRP`s income. The effect of pension contributions on historic income

#### Comparative Outputs:
##### Configuration A: Tuned Model (No RAG)
```text
When considering the payer-in-contribution (PIC) of an employee for their child or dependent care expenses under Child Maintenance Act 2019 section 37(4), it is necessary to treat any PICs as being made out of income and not part of the employer’s other earnings. The rules on how such deductions will be calculated can vary depending upon whether the worker has been paid in cash or through an arrangement where some amount may have already been deducted from that payment. However, if no deduction was taken at source then the calculation should take into account only what the individual would normally receive from the employers’ net profit after tax costs but before interest charges; this applies regardless of which method was used to calculate the cost figure provided. In addition, all allowances must still remain intact with regards to these amounts payable unless they could cause undue hardship on the taxpayer including if there were concerns about future claims arising during the period covered by them. If allowance figures do become too large than those previously allowed so as to exceed the permitted limit the excess cannot affect the entitlement to support payments for purposes mentioned above nor does its inclusion impact further steps needed to comply with the regulations relating to allowable amounts received. 

In determining appropriate amounts due to parents who make up their own benefits for their children, HMRC may consider both the benefit itself and any additional funds supplied by one or more persons involved in bringing up the child within the mean time between the date the benefit began and the
```

##### Configuration B: Base Model + RAG
```text
When calculating a paying parent's child maintenance liability under the Child Maintenance Act 1943, the amount paid by the payer towards maintaining the children is considered part of the payer's income. This means that any tax deductions made against the payers' income will also apply to the payments being made towards maintaining the children. Therefore, the net income available to the payer can only increase through regular payment of child maintenance. 

In other words, the payer must ensure that all taxes deductible from their income have been applied before making any further payments towards maintaining the children. Otherwise, there may be no additional funds left over after taking into account both employer and self-employed tax liabilities.
```

##### Configuration C: Tuned Model + RAG
```text
When calculating a paying parent's child maintenance liability, pension contributions will not affect the amount paid by the other parent. This means that the payment made by one parent can only cover what was earned during the year before the child left school, including any pensions received under the Child Benefit scheme. However, there may still be some tax implications related to pension contributions. For example, the employer contribution towards Pension Credit could potentially reduce the total allowance available for childcare payments. Therefore, the effective rate at which the child pays taxes should also take into account both the individual taxpayer's personal allowances and those set out in the Tax Credits Register. It is recommended that parents review their individual tax credits register regularly so that any changes do not result in unnecessary adjustments being made to their child maintenance liabilities.
```

---
